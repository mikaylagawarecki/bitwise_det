import torch
from torch.utils._pytree import tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode
import weakref
import math
import warnings
from contextlib import nullcontext
from collections.abc import Iterable

from torch.utils._mode_utils import no_dispatch

from torch.nn.attention import SDPBackend, sdpa_kernel

import traceback

class BitwiseEquivalenceMode(TorchDispatchMode):
    def __init__(self, raise_on_mismatch=False):
        """
        A TorchDispatchMode that runs every operator twice and compares the results for bitwise equivalence.
        
        Args:
            raise_on_mismatch (bool): Whether to raise an exception when a mismatch is detected
        """
        self.raise_on_mismatch = raise_on_mismatch
        self.mismatches = []
        self.matches = []
        self.changed_rng_state = []
        self.enabled = True
    
    def clone_inputs(self, args, kwargs):
        """
        Clone and detach all tensor inputs to avoid inplace modifications affecting the second run.
        """
        def clone_tensor(x):
            if x.requires_grad:
                return x.detach().clone()
            else:
                return x.detach().clone()
        
        cloned_args = tree_map_only(torch.Tensor, clone_tensor, args)
        cloned_kwargs = tree_map_only(torch.Tensor, clone_tensor, kwargs or {})
        return cloned_args, cloned_kwargs
    
    def compare_results(self, result1, result2, func_name):
        """
        Compare two results for bitwise equivalence.
        """
        def equal_with_nan(t1, t2):
            if torch.equal(t1, t2):
                return True

            mask1 = torch.isnan(t1)
            mask2 = torch.isnan(t2)

            # FIXME: avoid D2H here
            # Replace NaNs with a placeholder value (e.g., 0)
            tensor1_masked = torch.where(mask1, torch.tensor(0.0), t1)
            tensor2_masked = torch.where(mask2, torch.tensor(0.0), t2)

            # Check equality again after masking NaNs
            return torch.equal(tensor1_masked, tensor2_masked)

        def compare_tensor(t1, t2):
            if not equal_with_nan(t1, t2):
                mismatch_info = {
                    "function": func_name,
                    "tensor1_shape": t1.shape,
                    "tensor2_shape": t2.shape,
                    "max_diff": (t1 - t2).abs().max().item()
                    if t1.shape == t2.shape
                    else "Shape mismatch",
                }
                self.mismatches.append(mismatch_info)
                return False
            else:
                match_info = {
                    "function": func_name,
                    "tensor1_shape": t1.shape,
                    "tensor2_shape": t2.shape,
                    # "max_diff": (t1 - t2).abs().max().item() if t1.shape == t2.shape else "Shape mismatch"
                }
                self.matches.append(match_info)
            return True
        
        if isinstance(result1, torch.Tensor) and isinstance(result2, torch.Tensor):
            return compare_tensor(result1, result2)
        
        # figure out if I can use pytree for this
        def traverse_and_compare(obj1, obj2):
            if isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
                return compare_tensor(obj1, obj2)
            elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)) and len(obj1) == len(obj2):
                return all(traverse_and_compare(o1, o2) for o1, o2 in zip(obj1, obj2))
            elif isinstance(obj1, dict) and isinstance(obj2, dict) and obj1.keys() == obj2.keys():
                return all(traverse_and_compare(obj1[k], obj2[k]) for k in obj1)
            else:
                return True

        return traverse_and_compare(result1, result2)
    
    def __torch_dispatch__(self, func, types, func_args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        func_name = '{}.{}.{}'.format(*func._schema.name.split("::"), func._overloadname)
        # Clone inputs to avoid inplace modifications affecting the second run
        
        func_name = "{}.{}.{}".format(
            *func._schema.name.split("::"), func._overloadname
        )
        is_inplace = getattr(func._schema, "is_mutable", False)

        if "c10d" in func_name or "empty" in func_name:
            return func(*func_args, **kwargs)

        cloned_args, cloned_kwargs = func_args, kwargs
        # make inputs contiguous to avoid differences in reduction order when cloning with stride
        if not is_inplace:
            cloned_args, cloned_kwargs = func_args, kwargs
        else:
            func_args, kwargs = self.contiguous(func_args, kwargs, func)
            # Clone inputs to avoid inplace modifications affecting the second run
            cloned_args, cloned_kwargs = self.clone_inputs(func_args, kwargs)

        rng_state_before = torch.cuda.get_rng_state()
        # First run with original inputs
        result1 = func(*func_args, **kwargs)
        rng_state_after = torch.cuda.get_rng_state()

        if not torch.equal(rng_state_before, rng_state_after):
            torch.cuda.set_rng_state(rng_state_before)

        
        with torch.no_grad():
            result2 = func(*cloned_args, **cloned_kwargs)
        
        assert torch.equal(torch.cuda.get_rng_state(), rng_state_after), "RNG state should be unchanged after second run"
        
        # Compare results
        is_equal = self.compare_results(result1, result2, func_name)
        
        if not is_equal:
            message = f"Bitwise mismatch detected in {func_name}"
            if self.raise_on_mismatch:
                raise RuntimeError(message)
            else:
                warnings.warn(message)
        
        # Return the original result
        return result1
    
    def get_mismatches(self):
        """
        Get a list of all detected mismatches.
        """
        return self.mismatches


# Example usage:
if __name__ == "__main__":
    # Create a simple model
    import os
    for det in [True, False]:
        print(f"Running with det={det}")
        torch.use_deterministic_algorithms(det)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Create query, key, value tensors for attention
        B, M, N, nH, K, Kv = 1, 128, 1024, 8, 64, 64
        query = torch.randn(B, M, nH, K, dtype=torch.bfloat16, device='cuda', requires_grad=True).transpose(1, 2)
        key = torch.randn(B, N, nH, K, dtype=torch.bfloat16, device='cuda', requires_grad=True).transpose(1, 2)
        value = torch.randn(B, N, nH, Kv, dtype=torch.bfloat16, device='cuda', requires_grad=True).transpose(1, 2)

        with BitwiseEquivalenceMode(raise_on_mismatch=True) as mode:
            
            # This might not match due to non-determinism when dropout_p > 0
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.1, is_causal=False
                )
                attn_output.sum().backward()
        
            print(f"Detected {len(mode.get_mismatches())} mismatches with scaled_dot_product_attention")
