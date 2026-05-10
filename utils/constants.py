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
Constants used throughout the 4D Gaussian Splatting codebase.
Centralized location for all magic numbers and configuration values.
"""

# Random seed for reproducibility
DEFAULT_RANDOM_SEED = 6666

# Training parameters
DEFAULT_COARSE_ITERATIONS = 3000
DEFAULT_FINE_ITERATIONS = 14000
DEFAULT_ADAPTIVE_ITERATIONS = 14000

# Opacity and density thresholds
OPACITY_THRESHOLD_COARSE = 0.05
OPACITY_THRESHOLD_FINE_INIT = 0.05
OPACITY_THRESHOLD_ADAPTIVE_INIT = 0.03
OPACITY_THRESHOLD_FINE_AFTER = 0.05
OPACITY_THRESHOLD_ADAPTIVE_AFTER = 0.03

# Densification thresholds
DENSIFY_GRAD_THRESHOLD_COARSE = 0.0002
DENSIFY_GRAD_THRESHOLD_FINE_INIT = 0.0002
DENSIFY_GRAD_THRESHOLD_ADAPTIVE_INIT = 0.0001
DENSIFY_GRAD_THRESHOLD_FINE_AFTER = 0.0002
DENSIFY_GRAD_THRESHOLD_ADAPTIVE_AFTER = 0.0001

# Success threshold for anchor adjustment
SUCCESS_THRESHOLD = 0.8

# Memory management
MEMORY_CLEANUP_THRESHOLD = 1000  # Number of operations before cleanup

# Device configuration
DEFAULT_DEVICE = "cuda"
CUDA_DEVICE_ID = 0

# Image processing
IMAGE_SCALE_FACTOR = 255
IMAGE_CLAMP_MIN = 0
IMAGE_CLAMP_MAX = 1

# Network GUI
DEFAULT_GUI_IP = "127.0.0.1"
DEFAULT_GUI_PORT = 6009

# Video rendering
DEFAULT_VIDEO_FPS = 30

# File extensions
DEFAULT_IMAGE_EXTENSION = ".png"

# Gaussian model parameters
DEFAULT_FEAT_DIM = 32
DEFAULT_N_OFFSETS = 10
DEFAULT_VOXEL_SIZE = 0
DEFAULT_UPDATE_DEPTH = 3
DEFAULT_UPDATE_INIT_FACTOR = 16
DEFAULT_UPDATE_HIERARCHY_FACTOR = 4

# Logging intervals
PROGRESS_UPDATE_INTERVAL = 10
CHECKPOINT_SAVE_INTERVALS = [7000, 14000, 20000, 30000, 45000, 60000]
TEST_ITERATIONS = [7000, 14000, 20000, 30000, 45000, 60000]

# Training stages
STAGE_COARSE = "coarse"
STAGE_FINE = "fine"
STAGE_ADAPTIVE = "adaptive"

# Loss computation
EMA_ALPHA = 0.4
EMA_BETA = 0.6

# Multiprocessing
NUM_WORKERS = 32