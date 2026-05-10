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
"""4D Gaussian Splatting Rendering Script

This module implements the rendering pipeline for 4D Gaussian Splatting,
allowing for image and video generation from trained models.
"""

import concurrent.futures
import os
import threading
from argparse import ArgumentParser
from os import makedirs
from time import time
from typing import List, Tuple, Optional

import cv2
import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import render, prefilter_voxel, GaussianModel
from scene import Scene
from utils.constants import *
from utils.general_utils import safe_state
def multithread_write(image_list: List[torch.Tensor], path: str) -> None:
    """Write images using multithreading for improved performance.

    Args:
        image_list: List of image tensors to write
        path: Directory path to save images
    """
    def write_image(image: torch.Tensor, count: int, path: str) -> Tuple[int, bool]:
        """Write a single image to disk."""
        try:
            filename = os.path.join(path, f'{count:05d}.png')
            torchvision.utils.save_image(image, filename)
            return count, True
        except Exception as e:
            print(f"Failed to write image {count}: {e}")
            return count, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        tasks = [
            executor.submit(write_image, image, index, path)
            for index, image in enumerate(image_list)
        ]

        # Handle failed writes
        for task in concurrent.futures.as_completed(tasks):
            index, success = task.result()
            if not success:
                write_image(image_list[index], index, path)
    
# Utility function for converting tensors to 8-bit images
tensor_to_8bit = lambda tensor: (255 * np.clip(tensor.cpu().numpy(), 0, 1)).astype(np.uint8)
def render_set(model_path: str, name: str, iteration: int, views: List,
               gaussians: GaussianModel, pipeline, background: torch.Tensor,
               cam_type: str) -> None:
    """Render a set of views and save results.

    Args:
        model_path: Path to the trained model
        name: Name of the dataset/split being rendered
        iteration: Training iteration number
        views: List of camera views to render
        gaussians: Trained Gaussian model
        pipeline: Rendering pipeline parameters
        background: Background color tensor
        cam_type: Type of camera model used
    """
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians.get_gaussian_size)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        # breakpoint()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline,background)
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type,visible_mask=voxel_visible_mask)["render"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)
        # if idx >= 10:
            # break
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=DEFAULT_VIDEO_FPS)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(hyperparam,dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)