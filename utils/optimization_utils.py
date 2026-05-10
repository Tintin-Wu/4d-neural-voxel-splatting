"""
Optimization utilities for 4D Gaussian Splatting
Includes RMM memory pooling and efficient optimizers
"""

import torch
import warnings
from typing import Optional, Dict, Any

def setup_rmm_memory_pool(initial_pool_size: str = "2GiB",
                         maximum_pool_size: str = "8GiB") -> bool:
    """
    Setup RMM memory pooling for efficient GPU memory allocation.

    Args:
        initial_pool_size: Initial pool size (e.g., "2GiB", "1024MB")
        maximum_pool_size: Maximum pool size (e.g., "8GiB", "4096MB")

    Returns:
        bool: True if RMM was successfully configured, False otherwise
    """
    try:
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator

        # Setup memory pool
        pool = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=initial_pool_size,
            maximum_pool_size=maximum_pool_size
        )
        rmm.mr.set_current_device_resource(pool)

        # Configure PyTorch to use RMM
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

        print(f"✓ RMM memory pool configured: {initial_pool_size} initial, {maximum_pool_size} max")
        return True

    except ImportError:
        warnings.warn("RMM not available. Install with: pip install rmm-cu11")
        return False
    except Exception as e:
        warnings.warn(f"Failed to setup RMM: {e}")
        return False

def setup_efficient_optimizer(model_parameters,
                             optimizer_type: str = "adamw8bit",
                             offload_to_cpu: bool = False,
                             **optimizer_kwargs) -> torch.optim.Optimizer:
    """
    Setup memory-efficient optimizer.

    Args:
        model_parameters: Model parameters to optimize
        optimizer_type: Type of optimizer ("adamw8bit", "adamw4bit", "cpu_offload", "standard")
        offload_to_cpu: Whether to offload optimizer state to CPU
        **optimizer_kwargs: Additional optimizer arguments

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    try:
        if optimizer_type == "adamw8bit":
            from torchao.optim import AdamW8bit
            optimizer = AdamW8bit(
                model_parameters,
                bf16_stochastic_round=True,
                **optimizer_kwargs
            )
            print("✓ Using 8-bit AdamW optimizer (2x memory reduction)")

        elif optimizer_type == "adamw4bit":
            from torchao.optim import AdamW4bit
            optimizer = AdamW4bit(
                model_parameters,
                **optimizer_kwargs
            )
            print("✓ Using 4-bit AdamW optimizer (4x memory reduction)")

        elif optimizer_type == "cpu_offload" or offload_to_cpu:
            from torchao.optim import CPUOffloadOptimizer
            optimizer = CPUOffloadOptimizer(
                model_parameters,
                torch.optim.AdamW,
                offload_gradients=False,  # Keep False for gradient accumulation
                fused=True,
                **optimizer_kwargs
            )
            print("✓ Using CPU offload optimizer (extreme memory reduction)")

        else:
            # Standard PyTorch optimizer
            optimizer = torch.optim.Adam(
                model_parameters,
                **optimizer_kwargs
            )
            print("✓ Using standard Adam optimizer")

        return optimizer

    except ImportError:
        warnings.warn("TorchAO not available. Install with: pip install torchao")
        # Fallback to standard optimizer
        return torch.optim.Adam(model_parameters, **optimizer_kwargs)
    except Exception as e:
        warnings.warn(f"Failed to setup efficient optimizer: {e}")
        return torch.optim.Adam(model_parameters, **optimizer_kwargs)

def get_memory_stats() -> Dict[str, Any]:
    """Get current GPU memory statistics."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    return {}

def print_memory_usage(prefix: str = ""):
    """Print current GPU memory usage."""
    stats = get_memory_stats()
    if stats:
        print(f"{prefix}GPU Memory - Allocated: {stats['allocated']:.2f}GB, "
              f"Reserved: {stats['reserved']:.2f}GB, "
              f"Max: {stats['max_allocated']:.2f}GB")

def apply_quantization(model, quantization_type: str = "int8"):
    """
    Apply quantization to reduce model memory footprint.

    Args:
        model: PyTorch model to quantize
        quantization_type: Type of quantization ("int4", "int8")

    Returns:
        Quantized model
    """
    try:
        if quantization_type == "int4":
            from torchao.quantization import quantize_weights_wo_int4
            quantized_model = quantize_weights_wo_int4(model, group_size=64)
            print("✓ Applied 4-bit quantization (4x memory reduction)")

        elif quantization_type == "int8":
            from torchao.quantization import quantize_weights_wo_int8
            quantized_model = quantize_weights_wo_int8(model, group_size=32)
            print("✓ Applied 8-bit quantization (2x memory reduction)")

        else:
            quantized_model = model
            print("✓ No quantization applied")

        return quantized_model

    except ImportError:
        warnings.warn("TorchAO quantization not available. Install with: pip install torchao")
        return model
    except Exception as e:
        warnings.warn(f"Failed to apply quantization: {e}")
        return model