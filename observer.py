import logging

import os
import pickle
import sys
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from traceback import FrameSummary
from typing import Any, Dict, Optional

import torch

from hash_mode import HashMode
from torch import Tensor
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map_only

logger = logging.getLogger(__name__)

# Global observer instance
_observer = None

if not hasattr(torch, "xor_sum"):
    raise RuntimeError(
        "torch.xor_sum is not available, please patch https://github.com/pytorch/pytorch/pull/154149/"
    )


def hash_fn(t):
    with no_dispatch():
        if t.numel() == 0:
            return 0
        t = t.flatten()
        try:
            t = t.view(torch.int32)
        except RuntimeError as e:
            try:
                t = t.view(torch.uint8)
            except:
                t = t.contiguous().view(torch.uint8)
        return torch.xor_sum(t)


@dataclass
class Observation:
    name: str
    frames: list[FrameSummary]
    value: Any


@dataclass
class Run:
    """
    Represents a single run of a training job, aggregating multiple
    observations together.
    """

    name: str
    observations: list[Observation] = field(default_factory=list)
    # Store non-deterministic tensors separately
    non_deterministic_tensors: Dict[str, Any] = field(default_factory=dict)


class Observer:
    def __init__(
        self, run: str, reference_hash_file: str = None, raise_on_mismatch: bool = False
    ):
        self.run = Run(run)
        self.raise_on_mismatch = raise_on_mismatch
        self.mismatches = {}
        
        # Create a HashMode instance
        self.hash_mode = HashMode()
        self.hash_mode.register_observer(self)
        # Store gradient observation hook handles for cleanup
        self.grad_hooks = []
        # Dictionary to keep track of how many times each kind has been observed
        self.kind_counts = defaultdict(int)
        self.saved_tensors = {}

        # Load reference hashes if file is provided
        self.reference_hashes = []
        self.is_verification_run = reference_hash_file is not None
        if reference_hash_file is not None:
            self.reference_hashes, self.saved_tensors = self._read_reference_file(
                reference_hash_file
            )
            self.reference_hashes_idx = 0
    
    def _read_reference_file(
        self, reference_hash_file: str
    ) -> tuple[list[Observation], Dict[str, Any]]:
        """
        Read reference hashes and saved tensors from a file.

        Args:
            reference_hash_file (str): Path to the reference file

        Returns:
            tuple: (observations, non_deterministic_tensors)
        """
        observations = []
        non_deterministic_tensors = {}

        try:
            path = Path(reference_hash_file)
            if path.exists():
                logger.info(f"Loading reference hashes from {reference_hash_file}")
                with open(reference_hash_file, "rb") as f:
                    reference_run = pickle.load(f)
                    observations = reference_run.observations

                    # Load any saved tensors that have been marked as non-deterministic
                    non_deterministic_tensors = reference_run.non_deterministic_tensors
                    if non_deterministic_tensors:
                        logger.info(
                            f"Loaded {len(non_deterministic_tensors)} saved non-deterministic tensors"
                        )

                logger.info(f"Loaded {len(observations)} reference hashes")
        except Exception as e:
            logger.warning(f"Failed to load reference hashes: {e}")

        return observations, non_deterministic_tensors

    def _check_against_reference(self, unique_kind: str, hashed_value: Any) -> None:
        """
        Check a hashed value against reference hashes if available.

        Args:
            unique_kind (str): The unique name for this observation
            hashed_value (Any): The hashed value to check
        """
        if len(self.reference_hashes) == 0:
            return

        print(f"{unique_kind=}, {self.reference_hashes_idx=}")
        previous_observation = self.reference_hashes[self.reference_hashes_idx]
        reference_hash = previous_observation.value
        reference_kind = previous_observation.name

        # Check both the kind and the hash value
        if unique_kind != reference_kind:
            if self.raise_on_mismatch:
                raise RuntimeError(
                    f"Kind mismatch: current={unique_kind}, reference={reference_kind}"
                )
            else:
                logger.warning(
                    f"Kind mismatch: current={unique_kind}, reference={reference_kind}"
                )
                self.mismatches[unique_kind] = {
                    "current_kind": unique_kind,
                    "reference_kind": reference_kind,
                    "current_hash": hashed_value,
                    "reference_hash": reference_hash,
                }
        elif hashed_value != reference_hash:
            if self.raise_on_mismatch:
                raise RuntimeError(
                    f"Hash mismatch for {unique_kind}: current={hashed_value}, reference={reference_hash}, {previous_observation.frames}"
                )
            else:
                logger.warning(f"Hash mismatch for {unique_kind}")
                self.mismatches[unique_kind] = {
                    "current": hashed_value,
                    "reference": reference_hash,
                }
        else:
            logger.info(f"Hash match for {unique_kind}")

        self.reference_hashes_idx += 1

    def observe(self, kind: str, value: Any):
        """
        Record an observation of a given kind and value.

        This function creates a unique identifier for the observation, hashes the
        value, and logs the observation.

        It checks the hashed value against the reference hash if available.

        Args:
            kind (str): A name/identifier for this observation.
            value (Any): The value to be observed and hashed.
        """
        with no_dispatch():
            # Update the count for this kind
            self.kind_counts[kind] += 1

            # Create a unique name with the count
            unique_kind = f"{kind}_{self.kind_counts[kind]}"

            hashed_value = tree_map_only(Tensor, lambda x: hash_fn(x), value)
            logger.info(
                f"Hashed value {unique_kind} (original: {kind}): {hashed_value}"
            )

            # Check against reference hash if available
            self._check_against_reference(unique_kind, hashed_value)

            # Store the observation with the unique name
            self.run.observations.append(
                Observation(
                    unique_kind, frames=traceback.extract_stack(), value=hashed_value
                )
            )

    def observe_grad(self, kind: str, tensor: Tensor):
        """
        Register a hook on the tensor's gradient that will hash the gradient.

        Args:
            kind (str): A name/identifier for this observation
            tensor (Tensor): The tensor whose gradient will be observed
        """
        if not tensor.requires_grad:
            logger.warning(
                f"Cannot observe gradient for tensor that doesn't require grad: {kind}"
            )
            return

        # Update the count for this kind
        grad_kind = f"{kind}_grad"

        def hook(grad):
            with no_dispatch():
                logger.info(f"Observing gradient for {kind}")
                # The observe method will handle the indexing for us
                self.observe(grad_kind, grad)

        # Register the hook and store the handle for later removal
        handle = tensor.register_hook(hook)
        self.grad_hooks.append(handle)
        logger.info(f"Registered gradient observation hook for {kind}")

    def _remove_grad_hooks(self):
        """
        Remove all gradient observation hooks that were registered.
        This should be called when the Observer goes out of scope to prevent memory leaks.
        """
        if not self.grad_hooks:
            return

        logger.info(f"Removing {len(self.grad_hooks)} gradient observation hooks")
        for handle in self.grad_hooks:
            handle.remove()
        self.grad_hooks = []

    def mark_non_deterministic(self, name: str, tensor: Any):
        """
        Mark a tensor as non-deterministic. This tensor will be saved during the reference run
        and loaded during verification runs to ensure deterministic behavior.

        Args:
            name (str): A unique name for this tensor
            tensor (Any): The tensor to save (can be a single tensor or a PyTree of tensors)
        """
        # Update the count for this name
        self.kind_counts[name] += 1

        # Create a unique name with the count
        unique_name = f"{name}_{self.kind_counts[name]}"

        logger.info(
            f"Marking tensor '{unique_name}' (original: '{name}') as non-deterministic"
        )
        # Store the actual tensor value, not its hash
        self.run.non_deterministic_tensors[unique_name] = tensor

        # Return the unique name so it can be used for retrieval later
        return unique_name

    def _get_saved_tensor(self, name: str) -> Optional[Any]:
        """
        Retrieve a previously saved non-deterministic tensor.

        Args:
            name (str): The name of the tensor to retrieve. This should be the full name
                       with index already appended.

        Returns:
            The saved tensor if it exists, None otherwise
        """
        if name in self.saved_tensors:
            logger.info(f"Using saved tensor '{name}' from reference run")
            return self.saved_tensors[name]

        logger.warning(f"No saved tensor found for '{name}'")
        return None

    def enable_mode(self):
        """
        Enable the HashMode by pushing it onto the torch dispatch stack.
        This makes the mode active for all torch operations.
        """
        if self.hash_mode.enabled:
            logger.info("HashMode is already enabled")
            return

        logger.info("Enabling HashMode")
        # Push the mode onto the dispatch stack
        torch._C._push_on_torch_dispatch_stack(self.hash_mode)
        self.hash_mode.enabled = True
        logger.info("HashMode enabled")

    def disable_mode(self):
        """
        Disable the HashMode by finding and popping it from the torch dispatch stack.
        This prevents the mode from intercepting torch operations.
        """
        if not self.hash_mode.enabled:
            logger.info("HashMode is not currently enabled")
            return

        logger.info("Disabling HashMode")

        # Check if our HashMode instance is in the stack
        stack_len = torch._C._len_torch_dispatch_stack()
        found = False
        position = -1
        for i in range(stack_len):
            mode = torch._C._get_dispatch_stack_at(i)
            if mode is self.hash_mode:
                logger.info(f"Found HashMode at position {i} in the stack")
                found = True
                position = i
                break

        if found:
            # If HashMode is not at the top of the stack, we need special handling
            if position < stack_len - 1:
                raise RuntimeError(
                    "HashMode is not at the top of the stack, not yet implemented"
                )
                # FIXME: We need to properly handle torch.utils._python_dispatch._is_in_torch_dispatch_mode
                # and torch.utils._python_dispatch._is_in_non_infra_torch_dispatch_mode which are global
                # state that dynamo looks at when deciding whether to skip a frame or not

                # logger.warning(
                #     "HashMode is not at the top of the stack, using complex removal"
                # )
                # # Save all modes above HashMode
                # modes_to_restore = []
                # for i in range(stack_len - 1, position, -1):
                #     mode = torch._C._get_dispatch_stack_at(i)
                #     modes_to_restore.append(mode)
                #     logger.info(f"Saving {mode.__class__.__name__} at position {i}")
                #     torch._C._pop_torch_dispatch_stack(None)

                # # Now pop HashMode itself
                # torch._C._pop_torch_dispatch_stack(None)

                # # Restore the saved modes in reverse order to maintain original order
                # for mode in reversed(modes_to_restore):
                #     logger.info(f"Restoring {mode.__class__.__name__}")
                #     torch._C._push_on_torch_dispatch_stack(mode)
            else:
                # HashMode is at the top, just pop it
                torch._C._pop_torch_dispatch_stack(None)
                logger.info("HashMode was at the top of the stack and has been removed")

            self.hash_mode.enabled = False
            logger.info("HashMode successfully disabled")
        else:
            logger.warning("HashMode not found in the dispatch stack")
            self.hash_mode.enabled = False

    def report_mismatches(self):
        """Report any hash mismatches found during observation."""
        if not self.mismatches:
            logger.info("No hash mismatches found")
            return True

        logger.warning(f"Found {len(self.mismatches)} hash mismatches:")
        for kind, mismatch in self.mismatches.items():
            logger.warning(
                f"  {kind}: current={mismatch['current']}, reference={mismatch['reference']}"
            )
        return False

    def cleanup(self):
        """
        Clean up resources used by the Observer.
        This should be called when the Observer is no longer needed.
        """
        logger.info("Cleaning up Observer resources")
        self._remove_grad_hooks()
        self.disable_mode()


@contextmanager
def ObserverContext(
    dump_dir: str,
    name: str,
    reference_hash_file: str = None,
    raise_on_mismatch: bool = False,
):
    """
    Context manager for allowing observations.
    Args:
        dump_dir (str): The directory to dump the observer state to.
        name (str): The name of the run.
        reference_hash_file (str, optional): Path to a file containing reference hashes
            to compare against. If provided, observations will be checked against these
            reference hashes.
        raise_on_mismatch (bool, optional): Whether to raise an exception if a hash mismatch is found. Defaults to True.
    Yields:
        Observer: The current observer instance
    On exit, writes the observer state to the specified file and reports any hash mismatches.
    """
    global _observer

    path = Path(dump_dir) / f"{name}.pkl"
    filename = str(path)

    logger.info(f"Observations will be written to {filename}")

    # Create and set the global observer
    previous_observer = _observer
    _observer = Observer(name, reference_hash_file, raise_on_mismatch)

    try:
        yield _observer
    finally:
        # Clean up observer resources
        _observer.cleanup()

        # Report any hash mismatches
        if reference_hash_file is not None:
            _observer.report_mismatches()

        # Write the observer state to the file
        with open(filename, "wb") as f:
            pickle.dump(_observer.run, f)
        logger.info(f"Observations written to {filename}")

        # Restore the previous observer
        _observer = previous_observer


def observe(kind: str, value: Any):
    """
    Record a value to the global Observer. Does nothing if the global observer
    is not set.
    NOTE: this performs a host-device sync, so it should not be used in
    performance-critical code.
    """
    if _observer is not None:
        _observer.observe(kind, value)


def observe_grad(kind: str, tensor: Tensor):
    """
    Register a hook on the tensor's gradient that will save the hash of the gradient.

    Args:
        kind (str): A name/identifier for this observation
        tensor (Tensor): The tensor whose gradient will be observed

    """
    if _observer is not None:
        _observer.observe_grad(kind, tensor)


def mark_non_deterministic(name: str, tensor: Any):
    """
    Mark a tensor as non-deterministic. This tensor will be saved during the reference run
    and loaded during verification runs to ensure deterministic behavior.

    Args:
        name (str): A unique name for this tensor
        tensor (Any): The tensor to save (can be a single tensor or a PyTree of tensors)

    Returns:
        str: The unique name generated for this tensor (with index)
    """
    if _observer is not None:
        return _observer.mark_non_deterministic(name, tensor)
    return name  # Return the original name if no observer is active


def observe_non_deterministic(name: str, fn, *args, **kwargs):
    """
    Observe a non-deterministic operation. During reference runs, this will execute the function,
    mark its output as non-deterministic, and return the output. During verification runs, it will
    check if a saved tensor exists with the given name, and if so, return that instead of executing
    the function.

    Args:
        name (str): A unique name for this non-deterministic operation
        fn: The function to execute (if not using a saved tensor)
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function, or the saved tensor if available
    """
    if _observer is None:
        # If no observer is active, just run the function
        return fn(*args, **kwargs)

    with no_dispatch():
        if _observer.is_verification_run:
            # In verification mode, try to get the saved tensor
            saved_tensor = _observer._get_saved_tensor(name)
            if saved_tensor is not None:
                # Use the saved tensor instead of running the function
                return saved_tensor

        # Either in reference mode or no saved tensor available
        # Run the function and get its output
        output = fn(*args, **kwargs)

    # If in reference mode, mark this tensor as non-deterministic
    if not _observer.is_verification_run:
        # mark_non_deterministic now returns the unique name it generated
        _observer.mark_non_deterministic(name, output)

    return output


def enable_hash_mode():
    """
    Enable the HashMode on the global observer.
    """
    if _observer is not None:
        _observer.enable_mode()
    else:
        logger.warning("No global observer is set. Cannot enable HashMode.")


def disable_hash_mode():
    """
    Disable the HashMode on the global observer.
    """
    if _observer is not None:
        _observer.disable_mode()
    else:
        logger.warning("No global observer is set. Cannot disable HashMode.")
