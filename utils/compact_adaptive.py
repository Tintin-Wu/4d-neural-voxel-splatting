"""
Compact Adaptive Training - Memory-Efficient Implementation
Uses streaming statistics and lightweight sampling for large datasets
"""

import torch
import numpy as np
from typing import Dict, Optional, List
import random

class CompactQualityTracker:
    """Ultra-lightweight quality tracker using streaming statistics"""

    def __init__(self, ema_decay: float = 0.95, quality_threshold: float = 25.0):
        self.ema_decay = ema_decay
        self.quality_threshold = quality_threshold

        # Streaming statistics (no history storage)
        self.running_mean_psnr = 0.0
        self.running_std_psnr = 0.0
        self.count = 0

        # Current batch quality (updated each iteration)
        self.current_batch_quality = 0.0
        self.poor_view_ratio = 0.0

    def update_batch_quality(self, psnr_values: torch.Tensor) -> float:
        """Update with current batch PSNR values"""
        batch_psnr = psnr_values.mean().item()
        self.current_batch_quality = batch_psnr

        # Update streaming statistics
        self.count += 1
        if self.count == 1:
            self.running_mean_psnr = batch_psnr
            self.running_std_psnr = 0.0
        else:
            # EMA update
            delta = batch_psnr - self.running_mean_psnr
            self.running_mean_psnr += delta * (1 - self.ema_decay)
            self.running_std_psnr = self.ema_decay * self.running_std_psnr + (1 - self.ema_decay) * delta**2

        # Count poor views in batch
        poor_views = (psnr_values < self.quality_threshold).sum().item()
        self.poor_view_ratio = poor_views / len(psnr_values) if len(psnr_values) > 0 else 0.0

        return batch_psnr

    def get_lr_multiplier(self) -> float:
        """Get learning rate multiplier based on current quality"""
        if self.current_batch_quality < self.quality_threshold - 3:
            return 2.0  # Very poor quality
        elif self.current_batch_quality < self.quality_threshold:
            return 1.5  # Poor quality
        elif self.current_batch_quality > self.quality_threshold + 5:
            return 0.9  # Excellent quality
        else:
            return 1.0  # Normal quality

    def should_repeat_batch(self) -> bool:
        """Decide if current batch should be repeated (simple oversampling)"""
        return self.poor_view_ratio > 0.6  # If >60% of views are poor quality

class CompactAdaptiveSampler:
    """Lightweight importance sampling without storing full camera lists"""

    def __init__(self, total_cameras: int, batch_size: int, warmup_iterations: int = 500):
        self.total_cameras = total_cameras
        self.batch_size = batch_size
        self.warmup_iterations = warmup_iterations

        # Lightweight importance weights (only store for a subset)
        self.importance_indices = set()  # Track indices of important cameras
        self.recent_poor_indices = set()  # Recently poor performing cameras
        self.max_tracked_cameras = min(200, total_cameras // 10)  # Track at most 200 cameras

    def update_importance(self, camera_indices: List[int], psnr_values: torch.Tensor):
        """Update importance for current batch cameras"""
        poor_threshold = 25.0

        for i, (cam_idx, psnr) in enumerate(zip(camera_indices, psnr_values)):
            if psnr.item() < poor_threshold:
                self.recent_poor_indices.add(cam_idx)
                # Add to importance set
                if len(self.importance_indices) < self.max_tracked_cameras:
                    self.importance_indices.add(cam_idx)
            else:
                # Remove from poor indices if quality improved
                self.recent_poor_indices.discard(cam_idx)

    def sample_indices(self, iteration: int) -> List[int]:
        """Sample camera indices with importance bias"""
        if iteration < self.warmup_iterations or len(self.importance_indices) == 0:
            # Random sampling
            return random.sample(range(self.total_cameras), min(self.batch_size, self.total_cameras))

        # Biased sampling: 50% from important cameras, 50% random
        num_important = min(self.batch_size // 2, len(self.importance_indices))
        num_random = self.batch_size - num_important

        important_samples = random.sample(list(self.importance_indices), num_important)

        # Random samples (excluding already selected important ones)
        remaining_indices = list(set(range(self.total_cameras)) - set(important_samples))
        random_samples = random.sample(remaining_indices, min(num_random, len(remaining_indices)))

        return important_samples + random_samples

def apply_compact_adaptive_training(gaussians, iteration: int,
                                  viewpoint_cams: List,
                                  rendered_images: torch.Tensor,
                                  gt_images: torch.Tensor,
                                  quality_tracker: Optional[CompactQualityTracker] = None) -> Dict:
    """
    Apply compact adaptive training with minimal memory overhead

    Returns dict with adaptive training info
    """
    if quality_tracker is None:
        return {"lr_multiplier": 1.0, "repeat_batch": False, "quality": 0.0}

    # Compute PSNR for current batch
    from utils.image_utils import psnr
    psnr_values = []
    for i in range(len(viewpoint_cams)):
        if i < rendered_images.shape[0] and i < gt_images.shape[0]:
            psnr_val = psnr(rendered_images[i:i+1], gt_images[i:i+1])
            psnr_values.append(psnr_val)

    if not psnr_values:
        return {"lr_multiplier": 1.0, "repeat_batch": False, "quality": 0.0}

    psnr_tensor = torch.stack(psnr_values)

    # Update quality tracker
    avg_quality = quality_tracker.update_batch_quality(psnr_tensor)

    # Get adaptive training decisions
    lr_multiplier = quality_tracker.get_lr_multiplier()
    should_repeat = quality_tracker.should_repeat_batch()

    # Apply learning rate multiplier
    if hasattr(gaussians, 'optimizer') and lr_multiplier != 1.0:
        for param_group in gaussians.optimizer.param_groups:
            param_group['lr'] = param_group.get('base_lr', param_group['lr']) * lr_multiplier
            if 'base_lr' not in param_group:
                param_group['base_lr'] = param_group['lr'] / lr_multiplier

    return {
        "lr_multiplier": lr_multiplier,
        "repeat_batch": should_repeat,
        "quality": avg_quality,
        "poor_ratio": quality_tracker.poor_view_ratio,
        "running_mean": quality_tracker.running_mean_psnr
    }

# Integration helpers for existing codebase
def setup_compact_adaptive_training(dataset_size: int, batch_size: int,
                                   enable_adaptive: bool = True) -> tuple:
    """Setup compact adaptive training components"""
    if not enable_adaptive or dataset_size > 5000:  # Even more conservative limit
        return None, None

    quality_tracker = CompactQualityTracker()
    sampler = CompactAdaptiveSampler(dataset_size, batch_size)

    return quality_tracker, sampler

def log_compact_adaptive_stats(iteration: int, adaptive_info: Dict):
    """Log compact adaptive training statistics"""
    if iteration % 1000 == 0 and adaptive_info:
        print(f"Compact Adaptive [{iteration}]: "
              f"Quality={adaptive_info.get('quality', 0):.2f}, "
              f"LR_mult={adaptive_info.get('lr_multiplier', 1.0):.2f}, "
              f"Poor_ratio={adaptive_info.get('poor_ratio', 0):.1%}")