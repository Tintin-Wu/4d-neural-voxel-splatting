# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always use context7 and/or serena when generating code or looking up
library/API documentation — resolve library id and fetch library docs via the
Context7 MCP tools without needing an explicit ask.**


## Project Overview
4D Neural Voxel Splatting (ICIP 2026): an anchor-based neural-voxel formulation
of dynamic Gaussian splatting, combining the Scaffold-GS anchor representation
with multi-resolution HexPlane temporal deformation.

## Environment Setup

```bash
# 1. Python environment
conda create -n 4dnvs python=3.12 -y
conda activate 4dnvs

# 2. Install dependencies
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html

# 3. Build vendored CUDA extensions
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

The two `submodules/` directories are vendored, not git submodules — there is no
`git submodule update` step.

## Core Development Commands

### Training Commands
```bash
# Train on D-NeRF synthetic scenes
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py

# Train on DyNeRF real scenes (requires preprocessing)
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
bash colmap.sh data/dynerf/cut_roasted_beef llff
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py

# Train on HyperNeRF scenes
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
python train.py -s data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py
```

### Evaluation Commands
```bash
# Render images
python render.py --model_path "output/dnerf/bouncingballs/" --skip_train --configs arguments/dnerf/bouncingballs.py

# Compute metrics
python metrics.py --model_path "output/dnerf/bouncingballs/"
```

### Data Processing Commands
```bash
# Generate point clouds from multi-view images
bash colmap.sh <data_path> <datatype>  # datatype: blender, hypernerf, llff

# Downsample point clouds (recommended <40k points)
python scripts/downsample_point.py <input.ply> <output.ply>

# For DyNeRF: extract video frames
python scripts/preprocess_dynerf.py --datadir <dynerf_scene_path>
```

## Architecture Overview

### Core Components

**Scene Module (`scene/`)**
- `Scene`: Main scene container managing cameras, point clouds, and Gaussian models
- `GaussianModel`: Core 4D Gaussian representation with temporal deformation
- `dataset_readers.py`: Handles loading different dataset formats (D-NeRF, HyperNeRF, DyNeRF)
- `deformation.py`: Implements temporal deformation fields using HexPlane representations

**Rendering Pipeline (`gaussian_renderer/`)**
- `render()`: Main rendering function using differentiable Gaussian rasterization
- `generate_neural_gaussians()`: Generates time-dependent Gaussians from anchors
- Neural feature extraction and temporal interpolation

**Training Framework (`train.py`)**
- Two-stage training: coarse (3K iterations) → fine (30K iterations)
- Deformation-aware optimization with temporal consistency losses
- Adaptive pruning and densification strategies

### Key Data Structures

**4D Gaussian Representation**
- Anchor points with neural features (`_anchor_feat`)
- Temporal deformation networks for position/rotation/scale
- Multi-resolution HexPlane grids for efficient temporal modeling

**Dataset Support**
- **D-NeRF**: Synthetic scenes with known camera poses
- **HyperNeRF**: Real dynamic scenes with COLMAP reconstruction
- **DyNeRF**: Multi-camera real scenes requiring preprocessing
- **Multiple views**: Custom multi-camera setups

## Configuration System

All training parameters are defined in Python files under `arguments/`:
- `arguments/dnerf/`: D-NeRF dataset configurations
- `arguments/hypernerf/`: HyperNeRF dataset configurations
- `arguments/dynerf/`: DyNeRF dataset configurations

Each config file defines:
- `OptimizationParams`: Learning rates, iterations, pruning intervals
- `ModelParams`: Scene-specific parameters
- `PipelineParams`: Rendering pipeline settings

## Testing and Validation

No automated test suite. Validation through:
1. Training convergence monitoring via TensorBoard
2. Visual quality assessment of rendered outputs
3. Quantitative metrics (PSNR, SSIM, LPIPS) via `metrics.py`

## Important Notes

- CUDA 12.1 + PyTorch 2.2.0 required for the in-tree rasterization kernels
- Point cloud preprocessing critical - downsample to <40K points for memory efficiency
- Two-stage training essential: coarse stage establishes geometry, fine stage adds temporal details
- Port numbers in training commands prevent conflicts when running multiple experiments
- COLMAP integration required for real scene reconstruction