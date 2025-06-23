import logging

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
        "torch.xor_sum is not available, for now please patch https://github.com/pytorch/pytorch/pull/154149/"
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
        self, run: str, reference_hash_file: Optional[str] = None, raise_on_mismatch: bool = False
    ):
        self.run = Run(run)
        self.raise_on_mismatch = raise_on_mismatch
        self.mismatches = {}

        self.hash_mode = HashMode()
        self.hash_mode.register_observer(self)
        # Store gradient observation hook handles for cleanup
        self.grad_hooks = []
        # Gives id to dedup for when observe(kind) is called multiple times
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
                reference_run = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
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
            self.kind_counts[kind] += 1
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

    def observe_non_deterministic(self, name: str, tensor: Any):
        """
        Mark a tensor as non-deterministic. This tensor will be saved during the reference run
        and loaded during verification runs to ensure deterministic behavior.

        Args:
            name (str): A unique name for this tensor
            tensor (Any): The tensor to save (can be a single tensor or a PyTree of tensors)

        Returns:
            tensor: The original tensor or the saved tensor if this is a verification run
        """
        with no_dispatch():
            self.kind_counts[name] += 1
            unique_name = f"{name}_{self.kind_counts[name]}"

            if self.is_verification_run:
                tensor = self._get_saved_tensor(unique_name).to(tensor.device)
            else:
                logger.info(
                    f"Marking tensor '{unique_name}' (original: '{name}') as non-deterministic"
                )
                # Store the actual tensor value, not its hash
                self.run.non_deterministic_tensors[unique_name] = tensor
            
            return tensor

    def observe_non_deterministic_grad(self, name: str, tensor: Tensor):
        """
        Mark a tensor's gradient as non-deterministic. This registers a hook on the tensor
        that will save the gradient during the reference run and load it during verification runs.

        Args:
            name (str): A unique name for this gradient
            tensor (Tensor): The tensor whose gradient will be marked as non-deterministic

        Returns:
            str: The unique name that will be used for the gradient (with index)
        """
        if not tensor.requires_grad:
            logger.warning(
                f"Cannot mark gradient as non-deterministic for tensor that doesn't require grad: {name}"
            )
            return name

        # Create a name for the gradient
        grad_name = f"{name}_grad"

        def hook(grad):
            with no_dispatch():
                logger.info(f"Marking gradient for '{name}' as non-deterministic")
                # Use the existing mark_non_deterministic method to handle the gradient
                grad = self.observe_non_deterministic(grad_name, grad)
            return grad   

        # Register the hook and store the handle for later removal
        handle = tensor.register_hook(hook)
        self.grad_hooks.append(handle)
        logger.info(f"Registered non-deterministic gradient hook for '{name}'")

        # Return the name that will be used for the gradient
        return grad_name

    def _get_saved_tensor(self, name: str) -> Any:
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
        else:
            raise RuntimeError(f"Saved tensor '{name}' not found in reference run")

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
            torch.save(_observer.run, f)
        logger.info(f"Observations written to {filename}")

        # Restore the previous observer
        _observer = previous_observer


def observe(name: str, value: Any):
    """
    Record a hash of a tensor to the global Observer.

    NOTE: this performs a host-device sync, so it should not be used in
    performance-critical code.

    Args:
        name (str): A name/identifier for this observation
        tensor (Tensor): The tensor whose gradient will be observed
    """
    if _observer is not None:
        _observer.observe(name, value)


def observe_grad(name: str, tensor: Tensor):
    """
    Register a hook on the tensor's gradient that will save the hash of the
    gradient to the global observer.

    Args:
        name (str): A name/identifier for this observation
        tensor (Tensor): The tensor whose gradient will be observed

    """
    if _observer is not None:
        _observer.observe_grad(name, tensor)


def observe_non_deterministic(name: str, tensor: Any):
    """
    Mark a tensor as non-deterministic. This tensor will be saved during the reference run
    and loaded during verification runs to ensure deterministic behavior.

        Args:
            name (str): A unique name for this tensor
            tensor (Any): The tensor to save (can be a single tensor or a PyTree of tensors)

        Returns:
            tensor: The original tensor or the saved tensor if this is a verification run
    """
    if _observer is None:
        return tensor

    output = _observer.observe_non_deterministic(name, tensor)
    return output

def observe_non_deterministic_grad(name: str, tensor: Tensor):
    """
    Mark a tensor's gradient as non-deterministic. This registers a hook on the tensor
    that will save the gradient during the reference run and load it during verification runs.

    Args:
        name (str): A unique name for this gradient
        tensor (Tensor): The tensor whose gradient will be marked as non-deterministic

    Returns:
        str: The unique name that will be used for the gradient (with index)
    """
    if _observer is None:
        return tensor

    return _observer.observe_non_deterministic_grad(name, tensor)

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
