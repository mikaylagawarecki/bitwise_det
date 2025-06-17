#!/usr/bin/env python3
"""
Observer CI - Demonstrates using the Observer system for continuous integration
with expecttest-style tests.

This script shows how to:
1. Run a model and generate reference hashes
2. Run tests that verify the model's outputs against reference hashes
3. Update reference hashes when needed (similar to expecttest)

Usage:
  python observer_ci.py --generate  # Generate reference hashes
  python observer_ci.py --test      # Run tests against reference hashes
  python observer_ci.py --update    # Update reference hashes
"""

import argparse
import os
import random
from re import T
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our observer functionality
observer_path = os.path.dirname(os.path.abspath(__file__))
if observer_path not in sys.path:
    sys.path.insert(0, observer_path)
from observer import observe, observe_non_deterministic, ObserverContext


# Define a non-deterministic custom operator (similar to observer_demo.py)
class NonDeterministicOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        noise = torch.rand_like(input) * 0.01
        return input + noise

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        noise = torch.rand_like(grad_output) * 0.005
        return grad_output + noise


# Register the custom operator
non_deterministic_op = NonDeterministicOp.apply


# Create a simple model with non-deterministic operations
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)  # Dropout is non-deterministic
        self.fc = nn.Linear(
            16 * 32 * 32, 10
        )  # 32x32 is the feature map size after conv

    def forward(self, x):
        # First convolution + activation
        # Observe one of the weights in the conv layer
        observe("conv1_weight", self.conv1.weight)
        
        x = self.conv1(x)
        x = F.relu(x)
        observe("conv1_output", x)

        # Apply non-deterministic dropout
        # We use observe_non_deterministic to handle this operation
        x = observe_non_deterministic("dropout_output", self.dropout, x)
        observe("after_dropout", x)

        # Apply our custom non-deterministic op
        x = observe_non_deterministic(
            "custom_non_deterministic", non_deterministic_op, x
        )
        observe("after_custom_op", x)

        # Flatten and apply linear layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        observe("output", x)

        return x


# Constants
REFERENCE_DIR = "/tmp/observer_ci"
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "reference_run.pkl")
REFERENCE_FILE_HASH = os.path.join(REFERENCE_DIR, "reference_run_hash.pkl")


def generate_reference_hashes(model, batch_size=4):
    """Generate reference hashes for the model."""
    print("Generating reference hashes...")

    # Create the reference directory if it doesn't exist
    Path(REFERENCE_DIR).mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Generate input data
    input_data = torch.randn(batch_size, 3, 32, 32)

    # Use the observer context to record observations
    with ObserverContext(dump_dir=REFERENCE_DIR, name="reference_run") as observer:
        # Record the input
        observe("input", input_data)

        # Run the model
        output = model(input_data)
    
    # Use the observer context to record observations
    with ObserverContext(dump_dir=REFERENCE_DIR, name="reference_run_hash") as observer:
        observer.enable_mode()
        # Record the input
        observe("input", input_data)

        # Run the model
        output = model(input_data)

    print(f"Reference hashes saved to {REFERENCE_FILE}")
    return input_data


def run_test(model, input_data=None, batch_size=4, with_mode=False):
    """Run the model and compare against reference hashes."""
    print("Running test against reference hashes...")

    reference_file = REFERENCE_FILE_HASH if with_mode else REFERENCE_FILE
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Generate input data if not provided
    if input_data is None:
        input_data = torch.randn(batch_size, 3, 32, 32)

    # Use the observer context with reference hash file
    with ObserverContext(
        dump_dir=REFERENCE_DIR, name="test_run", reference_hash_file=reference_file
    ) as observer:
        if with_mode:
            observer.enable_mode()
        # Record the input
        observe("input", input_data)

        # Run the model
        output = model(input_data)

        # Check if there were any mismatches
        has_mismatches = len(observer.mismatches) > 0

    return not has_mismatches  # Return True if test passes (no mismatches)


class ObserverTests(unittest.TestCase):
    """Test cases using the Observer system."""

    def setUp(self):
        """Set up the test environment."""
        torch.manual_seed(42)
        self.model = SimpleModel()
        torch.manual_seed(42)
        self.reference_input = torch.randn(4, 3, 32, 32)

    def test_model_determinism(self):
        """Test that the model produces deterministic outputs with Observer."""
        # This test should pass if reference hashes exist and the model is deterministic
        # when using the Observer system
        self.assertTrue(
            run_test(self.model, self.reference_input),
            "Model outputs don't match reference hashes",
        )

    def test_different_input(self):
        """Test that different inputs produce different outputs."""
        # This test should fail because we're using different input data
        different_input = torch.randn(4, 3, 32, 32)  # Different random input
        self.assertFalse(
            run_test(self.model, different_input),
            "Model outputs match reference hashes with different input (unexpected)",
        )

    def test_without_observer(self):
        """Test the model without using the Observer system."""
        # Run the model twice with the same input but without Observer
        output1 = self.model(self.reference_input)

        output2 = self.model(self.reference_input)

        # These should be different due to non-deterministic operations
        # that aren't being handled by the Observer
        self.assertFalse(
            torch.allclose(output1, output2),
            "Model outputs match without Observer (unexpected)",
        )

    def test_with_hash_mode(self):
        """Test using the HashMode through enable_mode and disable_mode."""
        print("\nTesting with HashMode enabled...")

        self.assertTrue(
            run_test(self.model, self.reference_input, with_mode=True),
            "Model outputs don't match reference hashes",
        )


def update_reference_hashes():
    """Update reference hashes (similar to expecttest)."""
    print("Updating reference hashes...")

    # Simply regenerate the reference hashes
    torch.manual_seed(42)
    model = SimpleModel()
    generate_reference_hashes(model)

    print("Reference hashes updated.")


def main():
    """Main function to parse arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(description="Observer CI Demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate", action="store_true", help="Generate reference hashes"
    )
    group.add_argument(
        "--test", action="store_true", help="Run tests against reference hashes"
    )
    group.add_argument("--update", action="store_true", help="Update reference hashes")

    args = parser.parse_args()

    if args.generate:
        torch.manual_seed(42)
        model = SimpleModel()
        generate_reference_hashes(model)

    elif args.test:
        # Run the tests using unittest
        unittest.main(argv=[sys.argv[0]])

    elif args.update:
        update_reference_hashes()


if __name__ == "__main__":
    main()
