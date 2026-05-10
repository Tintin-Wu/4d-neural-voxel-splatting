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

"""Professional logging utilities for 4D Gaussian Splatting.

Provides structured logging, error handling, and progress tracking
for training and evaluation pipelines.
"""

import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Any, Dict, Union

import torch

def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_system_info(logger):
    """Log system and environment information."""
    import torch
    import platform

    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("CUDA: Not available")

def log_training_config(logger, args):
    """Log training configuration."""
    logger.info("Training Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

def log_model_info(logger, gaussians):
    """Log model information."""
    logger.info("Model Information:")
    logger.info(f"  Anchor points: {gaussians.get_anchor.shape[0]}")
    logger.info(f"  Total Gaussians: {gaussians.get_gaussian_size}")
    logger.info(f"  Feature dimension: {gaussians.feat_dim}")
    logger.info(f"  Number of offsets: {gaussians.n_offsets}")

class ProgressLogger:
    """Helper class for logging training progress."""

    def __init__(self, logger, log_interval=100):
        self.logger = logger
        self.log_interval = log_interval
        self.start_time = datetime.now()

    def log_iteration(self, iteration, stage, ema_loss, ema_psnr, total_points, num_anchors):
        """Log iteration progress."""
        if iteration % self.log_interval == 0:
            elapsed = datetime.now() - self.start_time
            self.logger.info(
                f"[{stage.upper()}] Iteration {iteration:05d} | "
                f"Loss: {ema_loss:.6f} | PSNR: {ema_psnr:.2f} | "
                f"Points: {total_points} | Anchors: {num_anchors} | "
                f"Elapsed: {elapsed}"
            )

    def log_stage_complete(self, stage, final_iteration):
        """Log stage completion."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Stage '{stage}' completed after {final_iteration} iterations in {elapsed}")

def create_training_logger(output_path, experiment_name):
    """Create a logger specifically for training."""
    log_file = os.path.join(output_path, f"{experiment_name}_training.log")
    return setup_logger('training', log_file)


@contextmanager
def error_handler(logger: logging.Logger, operation_name: str):
    """Context manager for graceful error handling with logging.

    Args:
        logger: Logger instance to use for error reporting
        operation_name: Name of the operation being performed

    Yields:
        None

    Raises:
        Re-raises the original exception after logging
    """
    try:
        logger.info(f"Starting {operation_name}")
        yield
        logger.info(f"Successfully completed {operation_name}")
    except Exception as e:
        logger.error(f"Error during {operation_name}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Log memory information if CUDA is available
        if torch.cuda.is_available():
            logger.error(f"GPU Memory: Allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                        f"Reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB")

        raise


class SafeOperationHandler:
    """Handler for operations that may fail but shouldn't crash the training."""

    def __init__(self, logger: logging.Logger, max_retries: int = 3):
        self.logger = logger
        self.max_retries = max_retries

    def safe_execute(self, operation: callable, operation_name: str,
                    *args, **kwargs) -> Optional[Any]:
        """Safely execute an operation with retries.

        Args:
            operation: Function to execute
            operation_name: Name for logging
            *args, **kwargs: Arguments to pass to operation

        Returns:
            Operation result or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {operation_name}: {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for {operation_name}")
                    return None

        return None


def log_training_metrics(logger: logging.Logger, iteration: int, metrics: Dict[str, float],
                        stage: str = "training") -> None:
    """Log training metrics in a structured format.

    Args:
        logger: Logger instance
        iteration: Current training iteration
        metrics: Dictionary of metric name -> value
        stage: Training stage (e.g., "coarse", "fine")
    """
    metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    logger.info(f"[{stage.upper()}] Iter {iteration:05d} | {metric_str}")


def log_memory_usage(logger: logging.Logger, context: str = "") -> None:
    """Log current memory usage.

    Args:
        logger: Logger instance
        context: Context string for the log message
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        prefix = f"[{context}] " if context else ""
        logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, "
                   f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


def handle_nan_loss(logger: logging.Logger, loss: torch.Tensor,
                   iteration: int) -> bool:
    """Handle NaN loss values gracefully.

    Args:
        logger: Logger instance
        loss: Loss tensor to check
        iteration: Current iteration

    Returns:
        True if loss is valid, False if NaN detected
    """
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"NaN/Inf loss detected at iteration {iteration}: {loss.item()}")
        logger.error("This may indicate:")
        logger.error("  - Learning rate too high")
        logger.error("  - Gradient explosion")
        logger.error("  - Numerical instability")
        log_memory_usage(logger, "NaN_Detection")
        return False
    return True