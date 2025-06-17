import logging

import torch
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)



class HashMode(TorchDispatchMode):
    def __init__(self):
        """
        A TorchDispatchMode that hashes outputs of each operation.
        """
        self.enabled = False
        self.observer = None
    
    def register_observer(self, observer):
        """Set the global observer reference for HashMode."""
        self.observer = observer

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        The main dispatch method that intercepts all PyTorch operations.

        Note: torch.compile will graph break on TorchDispatchMode, so we attempt
        to manually disable and enable the mode around the compiled region.

        Args:
            func: The PyTorch function being called
            types: The types of the arguments
            args: The positional arguments to the function
            kwargs: The keyword arguments to the function

        Returns:
            The result of calling the original function
        """

        # Call the original function
        result = func(*args, **(kwargs or {}))

        if self.observer is not None:
            # Use a consistent prefix for all observations from HashMode
            self.observer.observe(f"__torch_dispatch__{func.__name__}", result)

        return result
