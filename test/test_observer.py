#!/usr/bin/env python3
"""
Observer Tests - Demonstrates using the Observer system for continuous integration
with expecttest-style tests using assertExpectedInline.

This script shows how to:
1. Run a model and generate observations
2. Use assertExpectedInline to verify the model's outputs
3. Update expected outputs when needed (similar to expecttest)
"""

import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase

# Import our observer functionality
observer_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if observer_path not in sys.path:
    sys.path.insert(0, observer_path)
from observer import observe, observe_grad, observe_non_deterministic, ObserverContext


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


class ObserverTests(TestCase):
    """Test cases using the Observer system with assertExpectedInline."""

    def setUp(self):
        """Set up the test environment."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        self.model = SimpleModel()
        self.input_data = torch.randn(4, 3, 32, 32)
        self.maxDiff = None  # Show full diffs in test output

    def _load_observations(self, file_path):
        """Load observations from a pickle file and format them as a string."""
        with open(file_path, "rb") as f:
            run = pickle.load(f)
            result = ""
            for o in run.observations:
                result += f"{o.name} hash: {o.value}\n"
            return result

    def _run_model_and_get_observations(
        self, with_hash_mode=False, modified_input=None
    ):
        """Run the model and return the observations as a formatted string."""
        # Create a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run the model with observer
            with ObserverContext(dump_dir=temp_dir, name="test_run") as observer:
                if with_hash_mode:
                    observer.enable_mode()

                # Use the provided input or the default
                input_data = (
                    self.input_data if modified_input is None else modified_input
                )

                # Record the input
                observe("input", input_data)

                # Run the model
                output = self.model(input_data)

            # Load and return the observations
            return self._load_observations(os.path.join(temp_dir, "test_run.pkl"))

    def test_model_determinism(self):
        """Test that the model produces deterministic outputs with Observer."""
        # Run the model and get observations
        observations = self._run_model_and_get_observations()

        # Use assertExpectedInline to verify the expected observations
        self.assertExpectedInline(
            observations,
            """\
input_1 hash: 2136171533
conv1_weight_1 hash: 112767265
conv1_output_1 hash: 2030591017
after_dropout_1 hash: 90887988
after_custom_op_1 hash: 71213347
output_1 hash: -2115480567
""",
        )

    def test_different_input(self):
        """Test that different inputs produce different outputs."""
        # Run the model with the default input
        observations1 = self._run_model_and_get_observations()

        # Run the model with a different input
        different_input = torch.randn(4, 3, 32, 32)  # Different random input
        observations2 = self._run_model_and_get_observations(
            modified_input=different_input
        )

        # The observations should be different
        self.assertNotEqual(
            observations1,
            observations2,
            "Model outputs match with different input (unexpected)",
        )

    def test_without_observer(self):
        """Test the model without using the Observer system."""
        # Run the model twice with the same input but without Observer
        output1 = self.model(self.input_data)

        output2 = self.model(self.input_data)

        # These should be different due to non-deterministic operations
        # that aren't being handled by the Observer
        self.assertFalse(
            torch.equal(output1, output2),
            "Model outputs match without Observer (unexpected)",
        )

    def test_with_hash_mode(self):
        """Test using the HashMode through enable_mode and disable_mode."""
        # Run the model with HashMode enabled
        observations = self._run_model_and_get_observations(with_hash_mode=True)

        # Use assertExpectedInline to verify the expected observations
        self.assertExpectedInline(
            observations,
            """\
input_1 hash: 2136171533
conv1_weight_1 hash: 112767265
__torch_dispatch__convolution.default_1 hash: 186153873
__torch_dispatch__relu.default_1 hash: 2030591017
__torch_dispatch__detach.default_1 hash: 2030591017
conv1_output_1 hash: 2030591017
after_dropout_1 hash: 90887988
after_custom_op_1 hash: 71213347
__torch_dispatch__view.default_1 hash: 71213347
__torch_dispatch__t.default_1 hash: -2006178998
__torch_dispatch__addmm.default_1 hash: -2115480567
output_1 hash: -2115480567
""",
        )

    def test_observe_grad(self):
        """Test the observe_grad functionality."""
        # Create a simple model and input
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)

        # Create a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            # First run: generate reference hashes
            with ObserverContext(dump_dir=temp_dir, name="grad_test") as observer:
                # Register gradient observation on the weight parameter
                observe_grad("linear_weight_grad", model.weight)

                # Forward and backward pass
                output = model(input_tensor)
                loss = output.sum()
                loss.backward()

            # Load and format the observations
            observations = self._load_observations(
                os.path.join(temp_dir, "grad_test.pkl")
            )

            # Use assertExpectedInline to verify the expected observations
            self.assertExpectedInline(
                observations,
                """\
linear_weight_grad_grad_1 hash: -12740299
""",
            )

            # Second run: verify with modified input
            torch.manual_seed(43)  # Different seed to get different gradients
            modified_input = input_tensor * 1.1  # Scale the input to change gradients

            with ObserverContext(
                dump_dir=temp_dir, name="grad_test_modified"
            ) as observer:
                # Register gradient observation on the weight parameter
                observe_grad("linear_weight_grad", model.weight)

                # Forward and backward pass with modified input
                output = model(modified_input)
                loss = output.sum()
                loss.backward()

            # Load and format the observations
            modified_observations = self._load_observations(
                os.path.join(temp_dir, "grad_test_modified.pkl")
            )

            # The observations should be different
            self.assertNotEqual(
                observations,
                modified_observations,
                "Gradient observations match with modified input (unexpected)",
            )


if __name__ == "__main__":
    # Run the tests using PyTorch's test runner
    run_tests()
