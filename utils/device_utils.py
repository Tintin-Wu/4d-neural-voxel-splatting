#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
Device and tensor utilities for consistent device handling and memory management.
"""

import torch
import numpy as np
import random
from .constants import DEFAULT_DEVICE, DEFAULT_RANDOM_SEED

def get_device(device=None):
    """Get the appropriate device for tensor operations."""
    if device is None:
        device = DEFAULT_DEVICE

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"

    return device

def create_tensor_on_device(shape, dtype=torch.float32, device=None, fill_value=None):
    """Create a tensor on the specified device with optional fill value."""
    device = get_device(device)

    if fill_value is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    elif fill_value == 0:
        tensor = torch.zeros(shape, dtype=dtype, device=device)
    elif fill_value == 1:
        tensor = torch.ones(shape, dtype=dtype, device=device)
    else:
        tensor = torch.full(shape, fill_value, dtype=dtype, device=device)

    return tensor

def move_to_device(tensor_or_model, device=None):
    """Move tensor or model to specified device."""
    device = get_device(device)
    return tensor_or_model.to(device)

def setup_reproducible_training(seed=DEFAULT_RANDOM_SEED, device=None):
    """Set up reproducible training environment with proper seeding."""
    device = get_device(device)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(torch.device("cuda:0"))

    print(f"Reproducible training setup complete with seed {seed} on device {device}")

def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def safe_delete_tensors(*tensors):
    """Safely delete tensors and clean up memory."""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    cleanup_memory()

def get_tensor_memory_info():
    """Get current memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        return {
            "allocated": allocated / 1024**3,  # GB
            "cached": cached / 1024**3,        # GB
            "free": (torch.cuda.get_device_properties(0).total_memory - allocated) / 1024**3
        }
    return {"message": "CUDA not available"}