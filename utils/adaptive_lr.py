"""
Adaptive Learning Rate Controller for 4D Gaussian Splatting
Adjusts learning rates based on view quality to improve training efficiency
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class AdaptiveLearningRateController:
    """
    Controls learning rates based on view quality and training progress.
    Poor-quality views get higher learning rates to accelerate improvement.
    """

    def __init__(self,
                 base_learning_rates: Dict[str, float],
                 quality_threshold: float = 25.0,
                 max_lr_multiplier: float = 3.0,
                 min_lr_multiplier: float = 0.8,
                 smooth_transitions: bool = True,
                 transition_momentum: float = 0.9):
        """
        Initialize adaptive learning rate controller

        Args:
            base_learning_rates: Dictionary of parameter group names to base learning rates
            quality_threshold: PSNR threshold for determining poor quality
            max_lr_multiplier: Maximum learning rate multiplier for poor views
            min_lr_multiplier: Minimum learning rate multiplier for excellent views
            smooth_transitions: Whether to smooth learning rate transitions
            transition_momentum: Momentum for smooth transitions (0-1)
        """
        self.base_lrs = base_learning_rates.copy()
        self.quality_threshold = quality_threshold
        self.max_lr_multiplier = max_lr_multiplier
        self.min_lr_multiplier = min_lr_multiplier
        self.smooth_transitions = smooth_transitions
        self.transition_momentum = transition_momentum

        # Quality category definitions
        self.quality_categories = {
            'excellent': {'min_psnr': 28, 'multiplier': min_lr_multiplier},
            'good': {'min_psnr': 25, 'multiplier': 1.0},
            'fair': {'min_psnr': 22, 'multiplier': 1.5},
            'poor': {'min_psnr': 20, 'multiplier': 2.0},
            'very_poor': {'min_psnr': 0, 'multiplier': max_lr_multiplier}
        }

        # Parameter group specific multipliers (some parameters may need different treatment)
        self.param_specific_multipliers = {
            'anchor': 1.0,          # Position parameters
            'offset': 1.2,          # Offset parameters (slightly more aggressive)
            'mlp_opacity': 0.8,     # MLP parameters (more stable)
            'mlp_cov': 0.8,
            'mlp_color': 0.8,
            'deformation': 1.1,     # Deformation parameters
            'grid': 1.0,            # Grid parameters
            'default': 1.0          # Default for unspecified parameters
        }

        # Tracking for smooth transitions
        self.current_multipliers = {}
        self.target_multipliers = {}
        self.update_count = 0

        # Statistics tracking
        self.lr_history = {'iteration': [], 'multipliers': [], 'quality': []}
        self.quality_distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'very_poor': 0}

    def get_quality_category(self, psnr: float) -> str:
        """Categorize view quality based on PSNR"""
        for category, info in self.quality_categories.items():
            if psnr >= info['min_psnr']:
                return category
        return 'very_poor'

    def compute_lr_multiplier(self, current_view_quality: float,
                            param_group_name: str = 'default') -> float:
        """
        Compute learning rate multiplier for current view quality

        Args:
            current_view_quality: Current view PSNR
            param_group_name: Name of parameter group

        Returns:
            Learning rate multiplier
        """
        # Get base multiplier from quality category
        quality_category = self.get_quality_category(current_view_quality)
        base_multiplier = self.quality_categories[quality_category]['multiplier']

        # Apply parameter-specific adjustment
        param_multiplier = self.param_specific_multipliers.get(param_group_name, 1.0)

        # Compute final multiplier
        final_multiplier = base_multiplier * param_multiplier

        # Clamp to reasonable bounds
        final_multiplier = max(self.min_lr_multiplier,
                             min(self.max_lr_multiplier, final_multiplier))

        return final_multiplier

    def update_learning_rates(self,
                            optimizer: torch.optim.Optimizer,
                            current_view_quality: Optional[float] = None,
                            iteration: int = 0) -> Dict[str, float]:
        """
        Update optimizer learning rates based on current view quality

        Args:
            optimizer: PyTorch optimizer to update
            current_view_quality: Current view PSNR (None for standard scheduling)
            iteration: Current training iteration

        Returns:
            Dictionary of applied learning rate multipliers
        """
        applied_multipliers = {}

        if current_view_quality is None:
            # No adaptive adjustment, use base rates
            for param_group in optimizer.param_groups:
                param_name = param_group.get('name', 'default')
                applied_multipliers[param_name] = 1.0
            return applied_multipliers

        # Update quality distribution statistics
        quality_category = self.get_quality_category(current_view_quality)
        self.quality_distribution[quality_category] += 1

        # Compute target multipliers for each parameter group
        target_multipliers = {}
        for param_group in optimizer.param_groups:
            param_name = param_group.get('name', 'default')
            target_mult = self.compute_lr_multiplier(current_view_quality, param_name)
            target_multipliers[param_name] = target_mult

        # Apply smooth transitions if enabled
        if self.smooth_transitions:
            final_multipliers = self._smooth_transitions(target_multipliers)
        else:
            final_multipliers = target_multipliers

        # Apply multipliers to optimizer
        for param_group in optimizer.param_groups:
            param_name = param_group.get('name', 'default')
            base_lr = self.base_lrs.get(param_name, param_group.get('lr', 1e-4))

            multiplier = final_multipliers.get(param_name, 1.0)
            new_lr = base_lr * multiplier

            param_group['lr'] = new_lr
            applied_multipliers[param_name] = multiplier

        # Update tracking
        self.update_count += 1
        self._update_history(iteration, applied_multipliers, current_view_quality)

        return applied_multipliers

    def _smooth_transitions(self, target_multipliers: Dict[str, float]) -> Dict[str, float]:
        """Apply momentum-based smoothing to learning rate transitions"""
        final_multipliers = {}

        for param_name, target_mult in target_multipliers.items():
            if param_name not in self.current_multipliers:
                # Initialize
                self.current_multipliers[param_name] = 1.0

            # Apply exponential moving average
            current_mult = self.current_multipliers[param_name]
            smoothed_mult = (self.transition_momentum * current_mult +
                           (1 - self.transition_momentum) * target_mult)

            self.current_multipliers[param_name] = smoothed_mult
            final_multipliers[param_name] = smoothed_mult

        return final_multipliers

    def _update_history(self, iteration: int, multipliers: Dict[str, float], quality: float):
        """Update learning rate history for analysis"""
        self.lr_history['iteration'].append(iteration)
        self.lr_history['multipliers'].append(multipliers.copy())
        self.lr_history['quality'].append(quality)

        # Keep only recent history (last 1000 updates)
        max_history = 1000
        if len(self.lr_history['iteration']) > max_history:
            for key in self.lr_history:
                self.lr_history[key] = self.lr_history[key][-max_history:]

    def reset_base_learning_rates(self, new_base_rates: Dict[str, float]):
        """Update base learning rates (e.g., for different training stages)"""
        self.base_lrs.update(new_base_rates)

    def get_lr_statistics(self) -> Dict:
        """Get statistics about learning rate adaptations"""
        if not self.lr_history['iteration']:
            return {}

        # Compute statistics over recent history
        recent_multipliers = []
        recent_qualities = []

        for mult_dict, quality in zip(self.lr_history['multipliers'][-100:],
                                    self.lr_history['quality'][-100:]):
            # Average multiplier across parameter groups
            avg_mult = np.mean(list(mult_dict.values()))
            recent_multipliers.append(avg_mult)
            recent_qualities.append(quality)

        # Quality distribution percentages
        total_updates = sum(self.quality_distribution.values())
        quality_percentages = {}
        if total_updates > 0:
            for category, count in self.quality_distribution.items():
                quality_percentages[f'{category}_percent'] = 100 * count / total_updates

        return {
            'total_updates': self.update_count,
            'avg_multiplier_recent': np.mean(recent_multipliers) if recent_multipliers else 1.0,
            'std_multiplier_recent': np.std(recent_multipliers) if recent_multipliers else 0.0,
            'avg_quality_recent': np.mean(recent_qualities) if recent_qualities else 0.0,
            'current_multipliers': self.current_multipliers.copy(),
            **quality_percentages
        }

    def log_statistics(self, iteration: int):
        """Log learning rate adaptation statistics"""
        stats = self.get_lr_statistics()

        if stats and iteration % 1000 == 0:  # Log every 1000 iterations
            logging.info(f"Iteration {iteration} - Adaptive LR Stats:")
            logging.info(f"  Avg multiplier: {stats['avg_multiplier_recent']:.2f}")
            logging.info(f"  Recent quality: {stats['avg_quality_recent']:.2f}")
            logging.info(f"  Quality distribution: "
                        f"Excellent: {stats.get('excellent_percent', 0):.1f}%, "
                        f"Poor: {stats.get('poor_percent', 0):.1f}%")


class AdvancedLRScheduler:
    """
    Advanced learning rate scheduler that combines standard scheduling
    with adaptive quality-based adjustments
    """

    def __init__(self,
                 base_lr_controller: AdaptiveLearningRateController,
                 warmup_iterations: int = 500,
                 decay_schedule: Optional[Dict[int, float]] = None):
        """
        Initialize advanced LR scheduler

        Args:
            base_lr_controller: Base adaptive LR controller
            warmup_iterations: Number of warmup iterations
            decay_schedule: Optional decay schedule {iteration: decay_factor}
        """
        self.base_controller = base_lr_controller
        self.warmup_iterations = warmup_iterations
        self.decay_schedule = decay_schedule or {}

        # Warmup state
        self.warmup_start_lr = 0.1  # Start at 10% of base LR during warmup

    def step(self,
             optimizer: torch.optim.Optimizer,
             iteration: int,
             current_view_quality: Optional[float] = None) -> Dict[str, float]:
        """
        Perform learning rate update combining scheduling and adaptation

        Args:
            optimizer: PyTorch optimizer
            iteration: Current iteration
            current_view_quality: Current view quality for adaptation

        Returns:
            Applied learning rate multipliers
        """
        # Apply standard decay schedule first
        schedule_multiplier = 1.0
        for decay_iter, decay_factor in self.decay_schedule.items():
            if iteration >= decay_iter:
                schedule_multiplier = decay_factor

        # Apply warmup if in warmup phase
        if iteration < self.warmup_iterations:
            warmup_factor = self.warmup_start_lr + (1.0 - self.warmup_start_lr) * (iteration / self.warmup_iterations)
            schedule_multiplier *= warmup_factor

        # Update base learning rates with schedule
        scheduled_base_lrs = {}
        for param_name, base_lr in self.base_controller.base_lrs.items():
            scheduled_base_lrs[param_name] = base_lr * schedule_multiplier

        # Temporarily update base rates
        original_base_lrs = self.base_controller.base_lrs.copy()
        self.base_controller.reset_base_learning_rates(scheduled_base_lrs)

        # Apply adaptive adjustments
        multipliers = self.base_controller.update_learning_rates(
            optimizer, current_view_quality, iteration
        )

        # Restore original base rates
        self.base_controller.reset_base_learning_rates(original_base_lrs)

        return multipliers