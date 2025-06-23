#!/usr/bin/env python3

import os
import random
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_utils import run_tests, TestCase

# Import our observer functionality
observer_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if observer_path not in sys.path:
    sys.path.insert(0, observer_path)
from observer import (
    observe,
    observe_grad,
    observe_non_deterministic,
    observe_non_deterministic_grad,
    ObserverContext,
)


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


non_deterministic_op = NonDeterministicOp.apply


# Create a simple model with non-deterministic operations
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(
            16 * 32 * 32, 10
        )

    def forward(self, x):
        observe("conv1_weight", self.conv1.weight)

        x = self.conv1(x)
        x = F.relu(x)
        observe("conv1_output", x)

        x = self.dropout(x)
        observe_non_deterministic("after_dropout", x)

        x = non_deterministic_op(x)
        observe_non_deterministic("after_custom_op", x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        observe("output", x)

        return x


class SDPAModel(nn.Module):
    '''
    A simple model with an SDPA layer that uses the efficient attention backend
    for non-deterministic backward.
    '''
    def __init__(self, embed_dim=512, num_heads=8, head_dim=64, dropout_p=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p

        self.q_proj = nn.Linear(
            embed_dim, num_heads * head_dim, device="cuda", dtype=torch.bfloat16
        )
        self.k_proj = nn.Linear(
            embed_dim, num_heads * head_dim, device="cuda", dtype=torch.bfloat16
        )
        self.v_proj = nn.Linear(
            embed_dim, num_heads * head_dim, device="cuda", dtype=torch.bfloat16
        )
        self.out_proj = nn.Linear(
            num_heads * head_dim, embed_dim, device="cuda", dtype=torch.bfloat16
        )

    def forward(self, x_q, x_k, x_v):
        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p, is_causal=False
            )

        observe_non_deterministic_grad("q", q)
        observe_non_deterministic_grad("k", k)
        observe_non_deterministic_grad("v", v)

        attn_output = attn_output
        output = self.out_proj(attn_output)

        return output


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
        self.maxDiff = None

    def _load_observations(self, file_path):
        """Load observations from a pickle file and format them as a string."""
        run = torch.load(file_path, map_location="cpu", weights_only=False)
        result = ""
        for o in run.observations:
            result += f"{o.name} hash: {o.value}\n"
        return result

    def _run_model_and_get_observations(
        self, with_hash_mode=False, modified_input=None
    ):
        """Run the model and return the observations as a formatted string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ObserverContext(dump_dir=temp_dir, name="test_run") as observer:
                if with_hash_mode:
                    observer.enable_mode()

                input_data = (
                    self.input_data if modified_input is None else modified_input
                )

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
        observations = self._run_model_and_get_observations(with_hash_mode=True)

        self.assertExpectedInline(
            observations,
            """\
input_1 hash: 2136171533
conv1_weight_1 hash: 112767265
__torch_dispatch__convolution.default_1 hash: 186153873
__torch_dispatch__relu.default_1 hash: 2030591017
__torch_dispatch__detach.default_1 hash: 2030591017
conv1_output_1 hash: 2030591017
__torch_dispatch__empty_like.default_1 hash: 0
__torch_dispatch__bernoulli_.float_1 hash: 1065353216
__torch_dispatch__div_.Scalar_1 hash: 1067450368
__torch_dispatch__mul.Tensor_1 hash: 90887988
__torch_dispatch__rand_like.default_1 hash: 200463935
__torch_dispatch__mul.Tensor_2 hash: 98933601
__torch_dispatch__add.Tensor_1 hash: 71213347
__torch_dispatch__view.default_1 hash: 71213347
__torch_dispatch__t.default_1 hash: -2006178998
__torch_dispatch__addmm.default_1 hash: -2115480567
output_1 hash: -2115480567
""",
        )

    def test_observe_grad(self):
        """Test the observe_grad functionality."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)

        with tempfile.TemporaryDirectory() as temp_dir:
            with ObserverContext(dump_dir=temp_dir, name="grad_test") as observer:
                observe_grad("linear_weight_grad", model.weight)

                output = model(input_tensor)
                loss = output.sum()
                loss.backward()

            # Load and format the observations
            observations = self._load_observations(
                os.path.join(temp_dir, "grad_test.pkl")
            )

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


    def test_observe_non_deterministic_grad_sdpa(self):
        """Test the observe_non_deterministic_grad functionality with SDPA."""
        torch.use_deterministic_algorithms(False)
        torch.manual_seed(42)
        B, M, N, nH, K, Kv = 1, 128, 1024, 8, 64, 64
        model = SDPAModel(embed_dim=Kv, num_heads=nH, head_dim=K, dropout_p=0.1)

        query = torch.randn(
            B, M, nH, K, dtype=torch.bfloat16, device="cuda", requires_grad=True
        ).transpose(1, 2)
        key = torch.randn(
            B, N, nH, K, dtype=torch.bfloat16, device="cuda", requires_grad=True
        ).transpose(1, 2)
        value = torch.randn(
            B, N, nH, Kv, dtype=torch.bfloat16, device="cuda", requires_grad=True
        ).transpose(1, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            # First run: generate reference data with non-deterministic gradients
            reference_file = os.path.join(temp_dir, "sdpa_non_det_reference.pkl")

            with ObserverContext(
                dump_dir=temp_dir, name="sdpa_non_det_reference"
            ) as observer:
                observe_grad("q_proj_weight", model.q_proj.weight)

                # Forward pass through the model
                output = model(query, key, value)
                loss = output.sum()
                loss.backward()
            
            model.zero_grad()

            # Second run: verification run using the reference data
            with ObserverContext(
                dump_dir=temp_dir,
                name="sdpa_non_det_verification",
                reference_hash_file=reference_file,
            ) as observer:
                observe_grad("q_proj_weight", model.q_proj.weight)

                output = model(query, key, value)
                loss = output.sum()
                loss.backward()

                self.assertEqual(
                    len(observer.mismatches),
                    0,
                    f"Expected no mismatches but found: {observer.mismatches}",
                )

            model.zero_grad()
            
            # Third run: using a different seed to demonstrate that without the saved gradient,
            # we would get different results
            torch.manual_seed(43)

            with ObserverContext(
                dump_dir=temp_dir,
                name="sdpa_non_det_different_seed",
            ) as observer:

                observe_grad("q_proj_weight", model.q_proj.weight)

                # Forward pass through the model
                output = model(query, key, value)
                loss = output.sum()
                loss.backward()

            reference_observations = self._load_observations(reference_file)
            different_seed_observations = self._load_observations(
                os.path.join(temp_dir, "sdpa_non_det_different_seed.pkl")
            )

            self.assertNotEqual(
                reference_observations,
                different_seed_observations,
                "Observations match with different seed (unexpected)",
            )


if __name__ == "__main__":
    run_tests()
