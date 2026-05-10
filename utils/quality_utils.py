"""
Quality tracking utilities for adaptive training in 4D Gaussian Splatting
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.image_utils import psnr
from utils.loss_utils import ssim
import logging

class ViewQualityTracker:
    """Tracks and manages view quality metrics for adaptive training"""

    def __init__(self,
                 history_length: int = 10,
                 min_weight: float = 0.5,
                 max_weight: float = 3.0,
                 quality_threshold: float = 25.0):
        self.history_length = history_length
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.quality_threshold = quality_threshold

        # Storage for quality metrics
        self.view_quality_history: Dict[str, List[float]] = {}
        self.view_psnr_history: Dict[str, List[float]] = {}
        self.view_ssim_history: Dict[str, List[float]] = {}
        self.view_weights: Dict[str, float] = {}

        # Statistics
        self.global_quality_stats = {
            'mean_psnr': 0.0,
            'std_psnr': 0.0,
            'mean_quality': 0.0,
            'poor_view_count': 0,
            'total_views': 0
        }

    def compute_view_quality(self, rendered_image: torch.Tensor,
                           gt_image: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute quality metrics for a single view

        Args:
            rendered_image: Rendered image tensor [C, H, W]
            gt_image: Ground truth image tensor [C, H, W]

        Returns:
            Tuple of (combined_quality_score, psnr_value, ssim_value)
        """
        # Ensure tensors are properly shaped
        if rendered_image.dim() == 4:
            rendered_image = rendered_image.squeeze(0)
        if gt_image.dim() == 4:
            gt_image = gt_image.squeeze(0)

        # Compute PSNR and SSIM
        psnr_val = psnr(rendered_image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        ssim_val = ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0)).item()

        # Combined quality score (higher = better quality)
        # Weight PSNR more heavily as it's more stable
        quality_score = psnr_val * 0.8 + ssim_val * 20.0  # Normalize SSIM to similar scale

        return quality_score, psnr_val, ssim_val

    def update_view_quality(self, view_id: str,
                          rendered_image: torch.Tensor,
                          gt_image: torch.Tensor) -> float:
        """
        Update quality metrics for a specific view

        Args:
            view_id: Unique identifier for the view
            rendered_image: Rendered image tensor
            gt_image: Ground truth image tensor

        Returns:
            Current quality score for the view
        """
        quality_score, psnr_val, ssim_val = self.compute_view_quality(rendered_image, gt_image)

        # Initialize history if needed
        if view_id not in self.view_quality_history:
            self.view_quality_history[view_id] = []
            self.view_psnr_history[view_id] = []
            self.view_ssim_history[view_id] = []

        # Add to history with circular buffer behavior
        self.view_quality_history[view_id].append(quality_score)
        self.view_psnr_history[view_id].append(psnr_val)
        self.view_ssim_history[view_id].append(ssim_val)

        # Maintain history length
        if len(self.view_quality_history[view_id]) > self.history_length:
            self.view_quality_history[view_id].pop(0)
            self.view_psnr_history[view_id].pop(0)
            self.view_ssim_history[view_id].pop(0)

        return quality_score

    def get_view_average_quality(self, view_id: str) -> Optional[float]:
        """Get average quality for a specific view"""
        if view_id not in self.view_quality_history or not self.view_quality_history[view_id]:
            return None
        return np.mean(self.view_quality_history[view_id])

    def get_view_average_psnr(self, view_id: str) -> Optional[float]:
        """Get average PSNR for a specific view"""
        if view_id not in self.view_psnr_history or not self.view_psnr_history[view_id]:
            return None
        return np.mean(self.view_psnr_history[view_id])

    def compute_sampling_weights(self) -> Dict[str, float]:
        """
        Compute sampling weights based on view qualities

        Returns:
            Dictionary mapping view_id to sampling weight
        """
        if not self.view_quality_history:
            return {}

        # Get average qualities for all views
        avg_qualities = {}
        for view_id in self.view_quality_history:
            avg_quality = self.get_view_average_quality(view_id)
            if avg_quality is not None:
                avg_qualities[view_id] = avg_quality

        if not avg_qualities:
            return {}

        # Convert to numpy for easier computation
        view_ids = list(avg_qualities.keys())
        qualities = np.array(list(avg_qualities.values()))

        # Inverse relationship: poor quality = high weight
        # Add small epsilon to avoid division by zero
        inverse_qualities = 1.0 / (qualities + 1e-6)

        # Normalize weights to have mean = 1.0
        weights = inverse_qualities / inverse_qualities.mean()

        # Clamp weights to reasonable range
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Convert back to dictionary
        weight_dict = {view_id: float(weight) for view_id, weight in zip(view_ids, weights)}
        self.view_weights = weight_dict

        return weight_dict

    def update_global_stats(self):
        """Update global quality statistics"""
        all_psnr = []
        all_quality = []
        poor_count = 0

        for view_id in self.view_psnr_history:
            if self.view_psnr_history[view_id]:
                avg_psnr = np.mean(self.view_psnr_history[view_id])
                avg_quality = np.mean(self.view_quality_history[view_id])

                all_psnr.append(avg_psnr)
                all_quality.append(avg_quality)

                if avg_psnr < self.quality_threshold:
                    poor_count += 1

        if all_psnr:
            self.global_quality_stats.update({
                'mean_psnr': np.mean(all_psnr),
                'std_psnr': np.std(all_psnr),
                'mean_quality': np.mean(all_quality),
                'poor_view_count': poor_count,
                'total_views': len(all_psnr)
            })

    def get_quality_category(self, psnr: float) -> str:
        """Categorize view quality based on PSNR"""
        if psnr > 28: return 'excellent'
        elif psnr > 25: return 'good'
        elif psnr > 22: return 'fair'
        elif psnr > 20: return 'poor'
        else: return 'very_poor'

    def log_statistics(self, iteration: int):
        """Log quality statistics"""
        self.update_global_stats()
        stats = self.global_quality_stats

        if stats['total_views'] > 0:
            logging.info(f"Iteration {iteration} - Quality Stats:")
            logging.info(f"  Mean PSNR: {stats['mean_psnr']:.2f} ± {stats['std_psnr']:.2f}")
            logging.info(f"  Poor views: {stats['poor_view_count']}/{stats['total_views']} "
                        f"({100*stats['poor_view_count']/stats['total_views']:.1f}%)")

    def get_view_stats_summary(self) -> Dict:
        """Get summary of view statistics for logging"""
        self.update_global_stats()

        # Add weight statistics
        if self.view_weights:
            weights = list(self.view_weights.values())
            weight_stats = {
                'weight_mean': np.mean(weights),
                'weight_std': np.std(weights),
                'weight_min': np.min(weights),
                'weight_max': np.max(weights)
            }
        else:
            weight_stats = {}

        return {**self.global_quality_stats, **weight_stats}


def batch_compute_quality(rendered_images: torch.Tensor,
                         gt_images: torch.Tensor,
                         view_ids: List[str],
                         quality_tracker: ViewQualityTracker) -> List[float]:
    """
    Compute quality for a batch of images

    Args:
        rendered_images: Batch of rendered images [B, C, H, W]
        gt_images: Batch of ground truth images [B, C, H, W]
        view_ids: List of view identifiers
        quality_tracker: Quality tracker instance

    Returns:
        List of quality scores for each view
    """
    quality_scores = []

    for i, view_id in enumerate(view_ids):
        rendered = rendered_images[i] if rendered_images.dim() == 4 else rendered_images
        gt = gt_images[i] if gt_images.dim() == 4 else gt_images

        quality_score = quality_tracker.update_view_quality(view_id, rendered, gt)
        quality_scores.append(quality_score)

    return quality_scores