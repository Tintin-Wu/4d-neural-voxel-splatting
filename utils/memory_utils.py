"""
Memory management utilities for efficient tensor operations and GPU memory cleanup.
Provides context managers and utilities to reduce memory fragmentation and improve performance.
"""

import torch
import gc
from typing import Optional, Union, List, Any
from contextlib import contextmanager


class TensorManager:
    """Context manager for automatic tensor cleanup and memory management."""

    def __init__(self):
        self.tensors_to_cleanup = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up registered tensors and force garbage collection."""
        for tensor in self.tensors_to_cleanup:
            if tensor is not None:
                del tensor
        self.tensors_to_cleanup.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def register_for_cleanup(self, *tensors):
        """Register tensors for automatic cleanup when context exits."""
        self.tensors_to_cleanup.extend(tensors)


def efficient_tensor_cat(tensor1: torch.Tensor, tensor2: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Memory-efficient tensor concatenation that minimizes intermediate tensor creation.

    Args:
        tensor1: First tensor to concatenate
        tensor2: Second tensor to concatenate
        dim: Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    if tensor1 is None:
        return tensor2.clone()
    if tensor2 is None:
        return tensor1.clone()

    # Use torch.cat for simple concatenation - PyTorch optimizes this internally
    result = torch.cat([tensor1, tensor2], dim=dim)

    return result


def update_accumulator(accumulator: torch.Tensor, values: torch.Tensor,
                      indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Efficiently update accumulator tensor with new values.

    Args:
        accumulator: Existing accumulator tensor
        values: New values to add
        indices: Optional indices for scatter operations

    Returns:
        Updated accumulator tensor
    """
    if accumulator is None:
        return values.clone()

    if indices is not None:
        # Scatter add operation for sparse updates
        accumulator.scatter_add_(0, indices, values)
        return accumulator
    else:
        # Direct addition for dense updates
        if accumulator.shape[0] < values.shape[0]:
            # Expand accumulator if needed
            expansion_size = values.shape[0] - accumulator.shape[0]
            expansion_shape = list(accumulator.shape)
            expansion_shape[0] = expansion_size

            zeros = torch.zeros(expansion_shape, dtype=accumulator.dtype, device=accumulator.device)
            accumulator = torch.cat([accumulator, zeros], dim=0)

        accumulator[:values.shape[0]] += values
        return accumulator


class MemoryEfficientTensorList:
    """Smart tensor container that manages memory efficiently."""

    def __init__(self, initial_capacity: int = 1000):
        self.tensors = []
        self.capacity = initial_capacity

    def append(self, tensor: torch.Tensor):
        """Add tensor to the list."""
        self.tensors.append(tensor)

    def concatenate(self, dim: int = 0) -> torch.Tensor:
        """Concatenate all tensors in the list efficiently."""
        if not self.tensors:
            return None

        if len(self.tensors) == 1:
            return self.tensors[0]

        return torch.cat(self.tensors, dim=dim)

    def clear(self):
        """Clear all tensors and free memory."""
        for tensor in self.tensors:
            del tensor
        self.tensors.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def memory_efficient_operation():
    """Context manager for memory-efficient operations."""
    # Pre-operation cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    try:
        yield
    finally:
        # Post-operation cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    stats = {}

    if torch.cuda.is_available():
        stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
        stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB

    return stats


def print_memory_stats(prefix: str = ""):
    """Print current memory usage statistics."""
    stats = get_memory_usage()
    if stats:
        print(f"{prefix}GPU Memory - Allocated: {stats['gpu_allocated']:.2f}GB, "
              f"Reserved: {stats['gpu_reserved']:.2f}GB, "
              f"Max Allocated: {stats['gpu_max_allocated']:.2f}GB")