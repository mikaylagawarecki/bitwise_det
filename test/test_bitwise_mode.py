import functools
import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to sys.path to allow importing from determinism_tools
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from determinism_tools.bitwise_equivalence_mode import BitwiseEquivalenceMode
from torch.nn.attention import sdpa_kernel, SDPBackend


def deterministic_mode(enable_deterministic):
    """
    Decorator to set deterministic mode for a test function and restore original settings afterward.

    Args:
        enable_deterministic (bool): Whether to enable deterministic algorithms

    Returns:
        Decorated test function
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(self, *args, **kwargs):
            # Save original settings
            original_deterministic = torch.are_deterministic_algorithms_enabled()
            original_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)

            try:
                # Set deterministic settings
                torch.use_deterministic_algorithms(enable_deterministic)
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

                # Run the test function
                return test_func(self, *args, **kwargs)
            finally:
                # Restore original settings
                torch.use_deterministic_algorithms(original_deterministic)
                if original_cublas_config is not None:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_cublas_config
                elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
                    del os.environ["CUBLAS_WORKSPACE_CONFIG"]

        return wrapper

    return decorator


class TestBitwiseEquivalenceMode(unittest.TestCase):
    def setUp(self):
        # Common setup for both tests
        self.B, self.M, self.N, self.nH, self.K, self.Kv = 1, 128, 1024, 8, 64, 64

    def _create_attention_tensors(self):
        # Create query, key, value tensors for attention
        query = torch.randn(
            self.B,
            self.M,
            self.nH,
            self.K,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        ).transpose(1, 2)

        key = torch.randn(
            self.B,
            self.N,
            self.nH,
            self.K,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        ).transpose(1, 2)

        value = torch.randn(
            self.B,
            self.N,
            self.nH,
            self.Kv,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        ).transpose(1, 2)

        return query, key, value

    @deterministic_mode(True)
    def test_no_error_with_deterministic_algorithms(self):
        """Test that no error is raised when deterministic algorithms are set."""
        # Create tensors
        query, key, value = self._create_attention_tensors()

        # Run with BitwiseEquivalenceMode
        with BitwiseEquivalenceMode(raise_on_mismatch=True) as mode:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.1, is_causal=False
                )
                attn_output.sum().backward()

        # If we get here without an exception, the test passes
        self.assertEqual(
            len(mode.get_mismatches()),
            0,
            "Expected no mismatches with deterministic algorithms",
        )

    @deterministic_mode(False)
    def test_error_without_deterministic_algorithms(self):
        """Test that an error is raised when deterministic algorithms are not set."""
        # Create tensors
        query, key, value = self._create_attention_tensors()

        # Run with BitwiseEquivalenceMode and expect an exception
        with self.assertRaises(RuntimeError) as context:
            with BitwiseEquivalenceMode(raise_on_mismatch=True) as mode:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.1, is_causal=False
                    )
                    attn_output.sum().backward()

        # Check that the error message contains the expected function name
        self.assertIn(
            "aten._scaled_dot_product_efficient_attention_backward",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
