## Overview

Given an environment where **all** the software dependencies are frozen (torch version, library
dependencies of torch etc.) and **all** hardware is unchanged, this repository provides
1. Utilities to find sources of non-determinism in a PyTorch model
2. A framework for maintaining determinism of a PyTorch model over time

## Components

### BitwiseEquivalenceMode

A PyTorch dispatch mode that runs each operation twice and compares results for bitwise equivalence.

### HashMode

A PyTorch dispatch mode that hashes outputs of all operations registered to the
PyTorch dispatcher. This works with the Observer system to track and verify deterministic
behavior.

### Observer System

A framework for recording and comparing tensor hash values across different runs:

- Instrumentation points can be added via `observe`/`observe_grad`.
- HashMode can be enabled/disabled in regions via `{enable/disable}_mode()` to observe all ops in a region.
- Stores observations with stack traces for debugging
- Compares current hash values against reference values and reports mismatches between runs
- Allows marking tensors/ops as non-deterministic via `observe_non_deterministic()`. These tensors will be saved in the reference run and reloaded in subsequent runs, preventing the rest of the downstream hashes from mismatching.

## Usage

### BitwiseEquivalenceMode

```python
import torch
from bitwise_equivalence_mode import BitwiseEquivalenceMode

# Create a model and input
model = torch.nn.Linear(10, 5)
x = torch.randn(3, 10, requires_grad=True)

# Run with BitwiseEquivalenceMode
with BitwiseEquivalenceMode(raise_on_mismatch=False) as mode:
    y = model(x)
    loss = y.sum()
    loss.backward()

# Check for mismatches
print(f"Detected {len(mode.get_mismatches())} mismatches")
```

### HashMode

```python
import torch
from observer import ObserverContext

# Create an observer context which includes HashMode functionality
with ObserverContext(dump_dir="/tmp/hash_demo", name="hash_run") as observer:
    # Enable HashMode to start intercepting and hashing operations
    observer.enable_mode()

    # Run your model or operations
    model = torch.nn.Linear(10, 5)
    x = torch.randn(3, 10)
    output = model(x)

    # Disable HashMode when done
    observer.disable_mode()
```

### Observer System

```python
import torch
from observer import ObserverContext, observe

# Create a model
model = MyModel()

# Generate reference hashes
dump_dir = "/tmp/observer_demo"
with ObserverContext(dump_dir=dump_dir, name="reference_run") as observer:
    # Record input
    input_data = torch.randn(4, 3, 32, 32)
    observe("input", input_data)

    # Run model
    output = model(input_data)

    # Record output
    observe("output", output)

# Verify against reference hashes
reference_file = f"{dump_dir}/reference_run.pkl"
with ObserverContext(
    dump_dir=dump_dir,
    name="verification_run",
    reference_hash_file=reference_file
) as observer:
    # Same operations as before
    observe("input", input_data)
    output = model(input_data)
    observe("output", output)

    # Check for mismatches
    has_mismatches = len(observer.mismatches) > 0
```

## Demo

See `test/test_observer.py` for examples of using the Observer system.

