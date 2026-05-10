"""
Lightweight Quality Assessment for Identifying Crude Views
Memory-efficient alternatives to EMA-based tracking
"""

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import random

class LightweightQualityAssessment:
    """Ultra-lightweight crude view detection using minimal memory"""

    def __init__(self, quality_threshold: float = 25.0, percentile_threshold: float = 20.0):
        self.quality_threshold = quality_threshold
        self.percentile_threshold = percentile_threshold  # Bottom 20% are considered crude

        # Minimal storage - only recent samples
        self.recent_samples = {}  # camera_id -> last_psnr
        self.sample_count = 0
        self.running_stats = {'sum': 0, 'sum_sq': 0, 'count': 0}

    def update_sample(self, camera_id: str, psnr: float):
        """Update with single PSNR sample"""
        self.recent_samples[camera_id] = psnr

        # Update running statistics
        self.running_stats['sum'] += psnr
        self.running_stats['sum_sq'] += psnr * psnr
        self.running_stats['count'] += 1

        # Keep only recent N samples to limit memory
        if len(self.recent_samples) > 1000:
            # Remove oldest 20% of samples
            to_remove = list(self.recent_samples.keys())[:200]
            for key in to_remove:
                del self.recent_samples[key]

    def get_crude_views(self) -> Set[str]:
        """Get camera IDs of crude views using multiple criteria"""
        if len(self.recent_samples) < 10:
            return set()

        psnr_values = list(self.recent_samples.values())

        # Method 1: Absolute threshold
        crude_absolute = {cam_id for cam_id, psnr in self.recent_samples.items()
                         if psnr < self.quality_threshold}

        # Method 2: Percentile-based (bottom X%)
        percentile_cutoff = np.percentile(psnr_values, self.percentile_threshold)
        crude_percentile = {cam_id for cam_id, psnr in self.recent_samples.items()
                           if psnr <= percentile_cutoff}

        # Combine both methods
        return crude_absolute.union(crude_percentile)

    def get_quality_stats(self) -> Dict:
        """Get basic quality statistics"""
        if self.running_stats['count'] == 0:
            return {}

        mean_psnr = self.running_stats['sum'] / self.running_stats['count']
        var_psnr = (self.running_stats['sum_sq'] / self.running_stats['count']) - mean_psnr**2
        std_psnr = np.sqrt(max(0, var_psnr))

        return {
            'mean_psnr': mean_psnr,
            'std_psnr': std_psnr,
            'total_samples': self.running_stats['count'],
            'recent_samples': len(self.recent_samples)
        }

class PeriodicQualityCheck:
    """Periodic assessment without continuous tracking"""

    def __init__(self, check_interval: int = 500, sample_size: int = 100):
        self.check_interval = check_interval
        self.sample_size = sample_size
        self.last_check = 0
        self.crude_cameras = set()

    def should_check_quality(self, iteration: int) -> bool:
        """Decide if we should perform quality check this iteration"""
        return iteration - self.last_check >= self.check_interval

    def perform_quality_check(self, cameras: List, render_fn, iteration: int) -> Set[str]:
        """Perform one-time quality assessment on random sample"""
        if not self.should_check_quality(iteration):
            return self.crude_cameras

        # Sample random subset of cameras
        sample_cameras = random.sample(cameras, min(self.sample_size, len(cameras)))

        psnr_results = {}
        for cam in sample_cameras:
            try:
                # Render single view (implement this based on your rendering pipeline)
                psnr = self._compute_single_view_psnr(cam, render_fn)
                cam_id = getattr(cam, 'image_name', str(id(cam)))
                psnr_results[cam_id] = psnr
            except Exception as e:
                print(f"Failed to compute PSNR for camera: {e}")
                continue

        # Identify crude views
        if psnr_results:
            psnr_values = list(psnr_results.values())
            threshold = np.percentile(psnr_values, 25)  # Bottom 25%
            self.crude_cameras = {cam_id for cam_id, psnr in psnr_results.items()
                                 if psnr <= threshold or psnr < 22.0}

        self.last_check = iteration
        return self.crude_cameras

    def _compute_single_view_psnr(self, camera, render_fn) -> float:
        """Compute PSNR for single view - implement based on your pipeline"""
        # This is a placeholder - implement actual rendering
        # rendered_img = render_fn(camera)
        # gt_img = camera.original_image
        # return psnr(rendered_img, gt_img).item()
        return 25.0  # Placeholder

class GradientBasedQualityAssessment:
    """Use gradient information to identify crude views"""

    def __init__(self, grad_threshold: float = 0.0002):
        self.grad_threshold = grad_threshold
        self.camera_grad_history = defaultdict(list)
        self.max_history = 10

    def update_camera_gradients(self, camera_id: str, viewspace_grad: torch.Tensor):
        """Track gradient magnitude for camera"""
        grad_magnitude = viewspace_grad.norm().item()

        # Store recent gradient history
        self.camera_grad_history[camera_id].append(grad_magnitude)
        if len(self.camera_grad_history[camera_id]) > self.max_history:
            self.camera_grad_history[camera_id].pop(0)

    def get_high_gradient_cameras(self) -> Set[str]:
        """Get cameras with consistently high gradients (indicating poor convergence)"""
        crude_cameras = set()

        for cam_id, grad_history in self.camera_grad_history.items():
            if len(grad_history) >= 3:
                avg_grad = np.mean(grad_history)
                if avg_grad > self.grad_threshold:
                    crude_cameras.add(cam_id)

        return crude_cameras

class HybridLightweightDetector:
    """Combines multiple lightweight methods for robust crude view detection"""

    def __init__(self):
        self.quality_assessor = LightweightQualityAssessment()
        self.periodic_checker = PeriodicQualityCheck()
        self.gradient_assessor = GradientBasedQualityAssessment()

        self.final_crude_set = set()
        self.last_update = 0

    def update(self, camera_id: str, psnr: float, viewspace_grad: torch.Tensor = None):
        """Update all assessment methods"""
        self.quality_assessor.update_sample(camera_id, psnr)

        if viewspace_grad is not None:
            self.gradient_assessor.update_camera_gradients(camera_id, viewspace_grad)

    def get_crude_cameras(self, iteration: int, force_update: bool = False) -> Set[str]:
        """Get final set of crude cameras using all methods"""
        if not force_update and iteration - self.last_update < 100:
            return self.final_crude_set

        # Combine results from all methods
        quality_crude = self.quality_assessor.get_crude_views()
        gradient_crude = self.gradient_assessor.get_high_gradient_cameras()

        # Take intersection for high confidence, union for comprehensive coverage
        # You can tune this based on your needs
        self.final_crude_set = quality_crude.union(gradient_crude)

        self.last_update = iteration
        return self.final_crude_set

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        return {
            'quality_stats': self.quality_assessor.get_quality_stats(),
            'crude_count': len(self.final_crude_set),
            'total_tracked_cameras': len(self.quality_assessor.recent_samples),
            'gradient_tracked_cameras': len(self.gradient_assessor.camera_grad_history)
        }

# Integration functions for easy use in training loop
def setup_lightweight_quality_detection(method: str = "hybrid") -> object:
    """Setup lightweight quality detection method"""
    if method == "hybrid":
        return HybridLightweightDetector()
    elif method == "simple":
        return LightweightQualityAssessment()
    elif method == "periodic":
        return PeriodicQualityCheck()
    elif method == "gradient":
        return GradientBasedQualityAssessment()
    else:
        raise ValueError(f"Unknown method: {method}")

def update_quality_tracker(tracker, camera_id: str, psnr: float,
                          viewspace_grad: torch.Tensor = None):
    """Update quality tracker with current iteration data"""
    if hasattr(tracker, 'update'):
        tracker.update(camera_id, psnr, viewspace_grad)
    elif hasattr(tracker, 'update_sample'):
        tracker.update_sample(camera_id, psnr)

def get_crude_cameras_lightweight(tracker, iteration: int, cameras: List = None) -> Set[str]:
    """Get crude cameras using lightweight method"""
    if hasattr(tracker, 'get_crude_cameras'):
        return tracker.get_crude_cameras(iteration)
    elif hasattr(tracker, 'get_crude_views'):
        return tracker.get_crude_views()
    elif hasattr(tracker, 'perform_quality_check') and cameras:
        # For periodic checker
        return tracker.perform_quality_check(cameras, None, iteration)
    else:
        return set()