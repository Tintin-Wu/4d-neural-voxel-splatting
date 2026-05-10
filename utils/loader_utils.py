
import os
import cv2
import random
import numpy as np
from PIL import Image
 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
from typing import Dict, List, Optional, Iterator
import random
def get_stamp_list(dataset, timestamp):
    frame_length = int(len(dataset)/len(dataset.dataset.poses))
    # print(frame_length)
    if timestamp > frame_length:
        raise IndexError("input timestamp bigger than total timestamp.")
    print("select index:",[i*frame_length+timestamp for i in range(len(dataset.dataset.poses))])
    return [dataset[i*frame_length+timestamp] for i in range(len(dataset.dataset.poses))]
class FineSampler(Sampler):
    def __init__(self, dataset):
        self.len_dataset = len(dataset) 
        self.len_pose = len(dataset.dataset.poses)
        self.frame_length = int(self.len_dataset/ self.len_pose)

        sample_list = []
        for i in range(self.frame_length):
            for j in range(4):
                idx = torch.randperm(self.len_pose) *self.frame_length + i
                # print(idx)
                # breakpoint()
                now_list = []
                cnt = 0
                for item in idx.tolist():
                    now_list.append(item)
                    cnt+=1
                    if cnt % 2 == 0 and len(sample_list)>2:    
                        select_element = [x for x in random.sample(sample_list,2)]
                        now_list += select_element
            
            sample_list += now_list
            
        self.sample_list = sample_list
        # print(self.sample_list)
        # breakpoint()
        print("one epoch containing:",len(self.sample_list))
    def __iter__(self):

        return iter(self.sample_list)
    
    def __len__(self):
        return len(self.sample_list)


class QualityWeightedSampler:
    """
    Samples views based on their quality scores.
    Poor-quality views get higher sampling probability to improve training focus.
    """

    def __init__(self,
                 cameras: List,
                 batch_size: int = 2,
                 initial_weights: Optional[Dict[str, float]] = None,
                 fallback_to_random: bool = True,
                 warmup_iterations: int = 500):
        """
        Initialize quality-weighted sampler

        Args:
            cameras: List of camera objects
            batch_size: Number of cameras to sample per batch
            initial_weights: Initial sampling weights per camera
            fallback_to_random: Whether to fall back to random sampling if no weights
            warmup_iterations: Number of iterations before using weighted sampling
        """
        self.cameras = cameras
        self.batch_size = batch_size
        self.fallback_to_random = fallback_to_random
        self.warmup_iterations = warmup_iterations

        # Initialize camera identifiers
        self.camera_ids = [self._get_camera_id(cam, i) for i, cam in enumerate(cameras)]

        # Initialize weights
        if initial_weights:
            self.weights = self._align_weights_with_cameras(initial_weights)
        else:
            # Start with uniform weights
            self.weights = torch.ones(len(cameras))

        self.current_iteration = 0
        self.total_samples = 0
        self.poor_view_samples = 0

    def _get_camera_id(self, camera, index: int) -> str:
        """Generate unique identifier for a camera"""
        # Try to use camera's image name or time info if available
        if hasattr(camera, 'image_name') and camera.image_name:
            return f"cam_{camera.image_name}"
        elif hasattr(camera, 'time') and camera.time is not None:
            return f"cam_t{camera.time:.3f}_{index}"
        else:
            return f"cam_{index}"

    def _align_weights_with_cameras(self, weight_dict: Dict[str, float]) -> torch.Tensor:
        """Convert weight dictionary to tensor aligned with camera list"""
        weights = []
        for cam_id in self.camera_ids:
            if cam_id in weight_dict:
                weights.append(weight_dict[cam_id])
            else:
                weights.append(1.0)  # Default weight
        return torch.tensor(weights, dtype=torch.float32)

    def update_weights(self, quality_weights: Dict[str, float]):
        """
        Update sampling weights based on quality scores

        Args:
            quality_weights: Dictionary mapping camera IDs to weights
        """
        self.weights = self._align_weights_with_cameras(quality_weights)

        # Ensure weights are valid
        if torch.any(torch.isnan(self.weights)) or torch.any(self.weights <= 0):
            print("Warning: Invalid weights detected, falling back to uniform weights")
            self.weights = torch.ones(len(self.cameras))

    def sample_batch(self, iteration: int) -> List:
        """
        Sample a batch of cameras

        Args:
            iteration: Current training iteration

        Returns:
            List of sampled cameras
        """
        self.current_iteration = iteration

        # Use random sampling during warmup
        if iteration < self.warmup_iterations:
            return self._sample_random()

        # Use weighted sampling if weights are available
        if self.weights is not None and len(self.weights) == len(self.cameras):
            return self._sample_weighted()
        elif self.fallback_to_random:
            return self._sample_random()
        else:
            raise ValueError("No valid weights available and fallback disabled")

    def _sample_weighted(self) -> List:
        """Sample cameras using quality-based weights"""
        try:
            # Sample indices using multinomial distribution
            indices = torch.multinomial(
                self.weights,
                num_samples=min(self.batch_size, len(self.cameras)),
                replacement=True
            )

            sampled_cameras = [self.cameras[idx] for idx in indices]

            # Update statistics
            self.total_samples += len(sampled_cameras)

            # Count poor-quality views (assuming weight > 1.5 indicates poor quality)
            poor_indices = [idx for idx in indices if self.weights[idx] > 1.5]
            self.poor_view_samples += len(poor_indices)

            return sampled_cameras

        except Exception as e:
            print(f"Warning: Weighted sampling failed ({e}), falling back to random")
            return self._sample_random()

    def _sample_random(self) -> List:
        """Sample cameras randomly (fallback method)"""
        if len(self.cameras) <= self.batch_size:
            return self.cameras[:]

        indices = torch.randperm(len(self.cameras))[:self.batch_size]
        return [self.cameras[idx] for idx in indices]

    def get_sampling_stats(self) -> Dict:
        """Get statistics about sampling behavior"""
        if self.total_samples > 0:
            poor_view_ratio = self.poor_view_samples / self.total_samples
        else:
            poor_view_ratio = 0.0

        weight_stats = {}
        if self.weights is not None and len(self.weights) > 0:
            weight_stats = {
                'weight_mean': self.weights.mean().item(),
                'weight_std': self.weights.std().item(),
                'weight_min': self.weights.min().item(),
                'weight_max': self.weights.max().item(),
            }

        return {
            'total_samples': self.total_samples,
            'poor_view_samples': self.poor_view_samples,
            'poor_view_ratio': poor_view_ratio,
            'using_weighted_sampling': self.current_iteration >= self.warmup_iterations,
            **weight_stats
        }

    def reset_stats(self):
        """Reset sampling statistics"""
        self.total_samples = 0
        self.poor_view_samples = 0


class AdaptiveViewSampler:
    """
    Advanced sampler that combines temporal consistency with quality-based sampling
    """

    def __init__(self,
                 cameras: List,
                 batch_size: int = 2,
                 temporal_weight: float = 0.3,
                 quality_weight: float = 0.7):
        """
        Initialize adaptive view sampler

        Args:
            cameras: List of camera objects
            batch_size: Number of cameras to sample per batch
            temporal_weight: Weight for temporal consistency (0-1)
            quality_weight: Weight for quality-based sampling (0-1)
        """
        self.cameras = cameras
        self.batch_size = batch_size
        self.temporal_weight = temporal_weight
        self.quality_weight = quality_weight

        # Initialize time-based groups if temporal information is available
        self.temporal_groups = self._create_temporal_groups()

        # Track recently sampled views for temporal consistency
        self.recent_samples = []
        self.recent_history_length = 10

    def _create_temporal_groups(self) -> Dict[float, List[int]]:
        """Group cameras by timestamp for temporal consistency"""
        groups = {}

        for i, camera in enumerate(self.cameras):
            if hasattr(camera, 'time') and camera.time is not None:
                time_key = round(camera.time, 3)  # Round to avoid floating point issues
                if time_key not in groups:
                    groups[time_key] = []
                groups[time_key].append(i)

        return groups

    def sample_with_temporal_consistency(self,
                                       quality_weights: Optional[Dict[str, float]] = None,
                                       current_time: Optional[float] = None) -> List:
        """
        Sample cameras considering both quality and temporal consistency

        Args:
            quality_weights: Quality-based sampling weights
            current_time: Current timestamp for temporal consistency

        Returns:
            List of sampled cameras
        """
        if not quality_weights and not current_time:
            # Fall back to random sampling
            return torch.randperm(len(self.cameras))[:self.batch_size].tolist()

        # Combine quality and temporal sampling
        final_weights = torch.ones(len(self.cameras))

        # Apply quality weights
        if quality_weights:
            for i, camera in enumerate(self.cameras):
                cam_id = f"cam_{getattr(camera, 'image_name', i)}"
                if cam_id in quality_weights:
                    final_weights[i] *= quality_weights[cam_id] ** self.quality_weight

        # Apply temporal consistency
        if current_time and self.temporal_groups:
            temporal_boost = self._compute_temporal_weights(current_time)
            final_weights *= (temporal_boost ** self.temporal_weight)

        # Sample using combined weights
        try:
            indices = torch.multinomial(final_weights, self.batch_size, replacement=True)
            self.recent_samples.append(indices.tolist())

            # Maintain recent history
            if len(self.recent_samples) > self.recent_history_length:
                self.recent_samples.pop(0)

            return [self.cameras[idx] for idx in indices]
        except:
            # Fallback to random sampling
            indices = torch.randperm(len(self.cameras))[:self.batch_size]
            return [self.cameras[idx] for idx in indices]

    def _compute_temporal_weights(self, current_time: float) -> torch.Tensor:
        """Compute weights favoring temporally consistent views"""
        weights = torch.ones(len(self.cameras))

        for i, camera in enumerate(self.cameras):
            if hasattr(camera, 'time') and camera.time is not None:
                # Favor views close in time
                time_diff = abs(camera.time - current_time)
                # Exponential decay with time difference
                temporal_weight = torch.exp(-time_diff * 2.0)
                weights[i] = temporal_weight

        return weights
