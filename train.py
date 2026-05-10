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
"""4D Gaussian Splatting Training Script

This module implements the training pipeline for 4D Gaussian Splatting,
a method for real-time dynamic scene rendering using neural Gaussians.
"""

import copy
import os
import sys
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional, Dict, Any

import numpy as np
import torch
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render, network_gui, prefilter_voxel
from scene import Scene, GaussianModel
from scene.adaptive_visbility import AdaptiveVisibility
from utils.compact_adaptive import (
    CompactQualityTracker,
    apply_compact_adaptive_training,
    log_compact_adaptive_stats
)
from utils.constants import *
from utils.device_utils import setup_reproducible_training
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.lightweight_quality import (
    setup_lightweight_quality_detection,
    update_quality_tracker,
    get_crude_cameras_lightweight
)
from utils.loader_utils import FineSampler, get_stamp_list, QualityWeightedSampler
from utils.loss_utils import l1_loss, ssim
from utils.optimization_utils import setup_rmm_memory_pool, print_memory_usage
from utils.quality_utils import ViewQualityTracker, batch_compute_quality
from utils.scene_utils import render_training_image
from utils.timer import Timer

# Configure PIL for handling truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Utility function for converting tensors to 8-bit images
tensor_to_8bit = lambda tensor: (255 * np.clip(tensor.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    SummaryWriter = None
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer,adaptive_cam=None):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
    if stage == "fine":
        # gaussians.freeze()
        gaussians.reset_scheduler(opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    if stage == "adaptive":
        viewpoint_stack = adaptive_cam
        temp_list = copy.deepcopy(viewpoint_stack)
    elif not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size

    # Initialize adaptive training components with smart dataset size detection
    quality_tracker = None
    quality_sampler = None

    # Only enable adaptive training for reasonably sized datasets to avoid memory issues
    dataset_size = len(scene.getTrainCameras()) if hasattr(scene, 'getTrainCameras') else 0
    enable_adaptive = (hasattr(opt, 'enable_quality_weighting') and opt.enable_quality_weighting and
                      dataset_size < 2000)  # Limit to datasets with < 2000 cameras

    if enable_adaptive:
        print(f"Initializing adaptive training components for dataset with {dataset_size} cameras...")

        # Initialize quality tracker with reduced memory footprint
        quality_tracker = ViewQualityTracker(
            history_length=5,  # Reduced from 10 to save memory
            min_weight=getattr(opt, 'min_weight_multiplier', 0.5),
            max_weight=getattr(opt, 'max_weight_multiplier', 3.0),
            quality_threshold=getattr(opt, 'quality_threshold', 25.0)
        )

        # Initialize quality-weighted sampler only for non-adaptive stages
        if viewpoint_stack and stage != "adaptive":
            quality_sampler = QualityWeightedSampler(
                cameras=viewpoint_stack,
                batch_size=batch_size,
                warmup_iterations=getattr(opt, 'quality_warmup_iterations', 500)
            )

            # Note: Advanced temporal sampling available but not currently used

        print("✓ Adaptive training components initialized")
    elif dataset_size >= 2000:
        print(f"Dataset too large ({dataset_size} cameras) - adaptive training disabled to save memory")

    # Initialize compact adaptive training (minimal memory overhead)
    compact_quality_tracker = None
    if hasattr(opt, 'enable_compact_adaptive') and opt.enable_compact_adaptive and stage != "adaptive":
        compact_quality_tracker = CompactQualityTracker()
        print(f"✓ Compact adaptive training enabled for {dataset_size} cameras")

    # Initialize lightweight quality detection for crude view identification
    lightweight_detector = None
    if hasattr(opt, 'lightweight_quality_method') and stage in ["coarse", "fine"]:
        method = getattr(opt, 'lightweight_quality_method', 'hybrid')
        lightweight_detector = setup_lightweight_quality_detection(method)
        print(f"✓ Lightweight quality detection ({method}) initialized for crude view tracking")

    print("data loading done")
    if opt.dataloader:
        # For adaptive stage, use the already filtered viewpoint_stack (adaptive_cam)
        # For other stages, load all cameras (but only if dataset is not too large)
        if stage != "adaptive":
            viewpoint_stack = scene.getTrainCameras()

        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=32,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=32,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 

                             
    count = 0
    ema_grad = 0
    print(stage,": camera length",len(viewpoint_stack))
    for iteration in range(first_iter, final_iter+1):       
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(train_iter)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        # Standard learning rate update
        gaussians.update_learning_rate(iteration)

        # Track current view quality for adaptive LR (will be updated after rendering)
        current_view_quality = None

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch

        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=0,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            # Use adaptive sampling if available and not in adaptive stage
            if quality_sampler and stage != "adaptive" and iteration > quality_sampler.warmup_iterations:
                try:
                    viewpoint_cams = quality_sampler.sample_batch(iteration)
                except Exception as e:
                    print(f"Warning: Quality sampling failed, using random sampling: {e}")
                    viewpoint_cams = None
            else:
                viewpoint_cams = None

            # Fallback to random sampling
            if viewpoint_cams is None:
                idx = 0
                viewpoint_cams = []
                while idx < batch_size:
                    viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                    if not viewpoint_stack:
                        viewpoint_stack = temp_list.copy()
                    viewpoint_cams.append(viewpoint_cam)
                    idx += 1

            if len(viewpoint_cams) == 0:
                continue
            
        
        # print(len(viewpoint_cams))     
        # breakpoint()   
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        scaling_reg_list = []
        offset_indices_list = []
        voxel_visible_mask_list = []
        opacity_list = []
        
        for viewpoint_cam in viewpoint_cams:
            ## Scaffold_GS ##
            # if stage == "fine":
                # print(viewpoint_cam.time,viewpoint_cam.image_name)

            # Memory optimization: Clear cache if needed
            if hasattr(opt, 'empty_cache_frequency') and iteration % opt.empty_cache_frequency == 0:
                torch.cuda.empty_cache()

            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
            # print("voxel_visible_mask",voxel_visible_mask.shape)
            if voxel_visible_mask.sum() == 0:
                print("voxel_visible_mask is None, skip this camera")
                continue
            voxel_visible_mask_list.append(voxel_visible_mask)
            retain_grad = (iteration < opt.update_until and iteration >= 0)

            # Memory optimization: Use gradient checkpointing for large scenes
            if hasattr(opt, 'enable_memory_optimization') and opt.enable_memory_optimization:
                try:
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type,visible_mask=voxel_visible_mask, retain_grad=retain_grad)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Memory error at iteration {iteration}, clearing cache and retrying...")
                        torch.cuda.empty_cache()
                        # Retry with reduced precision if needed
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type,visible_mask=voxel_visible_mask, retain_grad=retain_grad)
                    else:
                        raise e
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type,visible_mask=voxel_visible_mask, retain_grad=retain_grad)
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            scaling = render_pkg["scaling"]
            opacity = render_pkg["opacity"]
            offset_indices = render_pkg["offset_indices"]
            scaling_reg_list.append(scaling.prod(dim=1).mean())
            offset_indices_list.append(offset_indices)
            opacity_list.append(opacity)
            ########################################
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.to('cuda')
            else:
                gt_image  = viewpoint_cam['image'].to('cuda')
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
            

        # Handle different tensor sizes in batch by taking element-wise max across all radii
        if len(radii_list) > 1:
            # Find the maximum size across all radii tensors
            max_size = max(r.size(1) for r in radii_list)
            # Pad smaller tensors and take max
            padded_radii = []
            for r in radii_list:
                if r.size(1) < max_size:
                    padding = torch.zeros(r.size(0), max_size - r.size(1), device=r.device, dtype=r.dtype)
                    r_padded = torch.cat([r, padding], dim=1)
                else:
                    r_padded = r
                padded_radii.append(r_padded)
            radii = torch.cat(padded_radii, 0).max(dim=0).values
        else:
            radii = radii_list[0].squeeze(0)

        # Handle visibility filter similarly
        if len(visibility_filter_list) > 1:
            max_size = max(v.size(1) for v in visibility_filter_list)
            padded_visibility = []
            for v in visibility_filter_list:
                if v.size(1) < max_size:
                    padding = torch.zeros(v.size(0), max_size - v.size(1), device=v.device, dtype=v.dtype)
                    v_padded = torch.cat([v, padding], dim=1)
                else:
                    v_padded = v
                padded_visibility.append(v_padded)
            visibility_filter = torch.cat(padded_visibility).any(dim=0)
        else:
            visibility_filter = visibility_filter_list[0].squeeze(0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        ## Scaffold_GS ##
        scaling_reg = sum(scaling_reg_list) / len(scaling_reg_list)
        # if stage == "coarse":
        offset_selection_mask = torch.cat(offset_selection_mask_list,0)
        ################
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        current_view_quality = psnr_.item()  # Store for adaptive LR

        # Update lightweight quality detector for crude view identification
        if lightweight_detector and len(viewpoint_cams) > 0:
            for i, cam in enumerate(viewpoint_cams):
                cam_id = getattr(cam, 'image_name', f'cam_{i}')
                # Get per-camera PSNR if batch
                if len(viewpoint_cams) > 1 and i < len(image_tensor):
                    cam_psnr = psnr(image_tensor[i:i+1], gt_image_tensor[i:i+1]).item()
                else:
                    cam_psnr = psnr_.item()

                # Update quality tracker with PSNR and gradient info
                # Get viewspace gradient for this camera (if available)
                if i < len(viewspace_point_tensor_list) and hasattr(viewspace_point_tensor_list[i], 'grad') and viewspace_point_tensor_list[i].grad is not None:
                    viewspace_grad = viewspace_point_tensor_list[i]
                else:
                    viewspace_grad = None
                update_quality_tracker(lightweight_detector, cam_id, cam_psnr, viewspace_grad)

        # Update quality tracking and sampling weights
        if quality_tracker and len(viewpoint_cams) > 0:
            try:
                # Track quality for each view in the batch
                batch_compute_quality(
                    image_tensor, gt_image_tensor,
                    [f"cam_{getattr(cam, 'image_name', i)}" for i, cam in enumerate(viewpoint_cams)],
                    quality_tracker
                )

                # Update sampling weights periodically
                if iteration % getattr(opt, 'quality_update_interval', 100) == 0:
                    quality_weights = quality_tracker.compute_sampling_weights()
                    if quality_sampler and quality_weights:
                        quality_sampler.update_weights(quality_weights)

                    # Log quality statistics
                    if iteration % 1000 == 0:
                        quality_tracker.log_statistics(iteration)

            except Exception as e:
                print(f"Warning: Quality tracking failed: {e}")

        # Apply compact adaptive training (minimal memory overhead)
        adaptive_info = {}
        if compact_quality_tracker and iteration > getattr(opt, 'compact_adaptive_warmup', 1000):
            try:
                adaptive_info = apply_compact_adaptive_training(
                    gaussians, iteration, viewpoint_cams,
                    image_tensor, gt_image_tensor, compact_quality_tracker
                )
                # Log statistics
                log_compact_adaptive_stats(iteration, adaptive_info)
            except Exception as e:
                print(f"Warning: Compact adaptive training failed: {e}")

        # norm
        loss = Ll1
        if (stage == "fine" or stage=="adaptive") and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_scaling_reg != 0 :
            loss += opt.lambda_scaling_reg * scaling_reg
        loss.backward()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("NaN loss detected. Attempting recovery...")
            # Instead of restarting, try to recover by reducing learning rates
            for param_group in gaussians.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Reduced learning rates by 50%. Continuing training...")
            continue

        # Apply adaptive learning rate adjustments if enabled
        if (hasattr(opt, 'enable_adaptive_lr') and opt.enable_adaptive_lr and
            hasattr(gaussians, 'update_learning_rate_adaptive')):
            try:
                gaussians.update_learning_rate_adaptive(iteration, current_view_quality)
            except Exception as e:
                print(f"Warning: Adaptive LR update failed: {e}")
            
        
        iter_end.record()

        
        
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = EMA_ALPHA * loss.item() + EMA_BETA * ema_loss_for_log
            ema_psnr_for_log = EMA_ALPHA * psnr_ + EMA_BETA * ema_psnr_for_log
            total_point = gaussians.get_gaussian_size
            if iteration % PROGRESS_UPDATE_INTERVAL == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}",
                                          "anchor":f"{gaussians.get_anchor.shape[0]}"},)
                progress_bar.update(PROGRESS_UPDATE_INTERVAL)
            if iteration == train_iter:
                progress_bar.close()
            
            # Lightweight adaptive camera detection (replaces EMA-based logic)
            if stage == "fine" and iteration > opt.add_adaptive_cam_from_iter and iteration % opt.adaptive_per_iter == 0:
                if lightweight_detector:
                    # Get crude cameras using lightweight detection
                    crude_camera_ids = get_crude_cameras_lightweight(lightweight_detector, iteration)

                    # Convert camera IDs back to camera objects and add to adaptive_cam
                    for cam in viewpoint_cams:
                        cam_id = getattr(cam, 'image_name', f'cam_{id(cam)}')
                        if cam_id in crude_camera_ids and cam not in adaptive_cam:
                            adaptive_cam.append(cam)
                            print(f"Added crude camera {cam_id} to adaptive_cam, total length = {len(adaptive_cam)}")

                    # Log quality statistics periodically
                    if iteration % 1000 == 0 and hasattr(lightweight_detector, 'get_summary_stats'):
                        stats = lightweight_detector.get_summary_stats()
                        print(f"Quality Stats [iter {iteration}]: {stats}")

                else:
                    # Fallback to original EMA-based logic if lightweight detector not available
                    threshold_multiplier = opt.ema_grad_offset_init + iteration * (opt.ema_grad_offset_final - opt.ema_grad_offset_init) / opt.densify_until_iter
                    if opt.adaptive_strat == "grad":
                        viewspace_point_tensor_grad = viewspace_point_tensor.grad
                        if viewspace_point_tensor_grad is not None:
                            viewspace_point_tensor_grad = viewspace_point_tensor_grad.detach().norm()
                            if iteration == opt.add_adaptive_cam_from_iter:
                                ema_grad = viewspace_point_tensor_grad

                            ema_grad = 0.4*viewspace_point_tensor_grad + 0.6*ema_grad

                            if viewspace_point_tensor_grad >(1+threshold_multiplier)*ema_grad:
                                if viewpoint_cams[0] not in adaptive_cam:
                                    adaptive_cam.append(viewpoint_cams[0])
                                    print("add adaptive cam, length = ", len(adaptive_cam))
                    elif opt.adaptive_strat == "psnr":
                        if psnr_ > (1+threshold_multiplier)*ema_psnr_for_log and viewpoint_cams[0] not in adaptive_cam:
                            adaptive_cam.append(viewpoint_cams[0])
                    else:
                        raise NotImplementedError("adaptive_strat not implemented")
                
            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                ## Scaffold_GS ##
                #################   
                success_threshold = opt.success_threshold
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                elif stage == "fine":
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_fine_after)/(opt.densify_until_iter )
                else:
                    opacity_threshold = opt.opacity_threshold_adaptive_init - iteration*(opt.opacity_threshold_adaptive_init - opt.opacity_threshold_adaptive_after)/(opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_adaptive_init - iteration*(opt.densify_grad_threshold_adaptive_init - opt.densify_grad_threshold_adaptive_after)/(opt.densify_until_iter )

                # TODO: stats only use the first batch item; extend to accumulate across the full batch.
                if viewspace_point_tensor_list:
                    gaussians.training_statis(
                        viewspace_point_tensor_list[0],
                        opacity_list[0],
                        visibility_filter_list[0].squeeze(0),
                        offset_indices_list[0],
                        voxel_visible_mask_list[0],
                    )

                if iteration > opt.update_from and iteration % opt.update_interval == 0 :
                    gaussians.adjust_anchor(extent=scene.cameras_extent, check_interval=opt.update_interval, success_threshold=success_threshold, grad_threshold=densify_threshold, min_opacity=opacity_threshold)

                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
                # if iteration % opt.opacity_reset_interval == 0:
                #     print("reset opacity")
                #     gaussians.reset_opacity()
                    

            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # Setup memory optimizations
    print("=" * 50)
    print("Initializing Memory Optimizations")
    print("=" * 50)

    # Setup RMM memory pooling
    if opt.use_rmm_pool:
        setup_rmm_memory_pool(
            initial_pool_size="2GiB",
            maximum_pool_size="8GiB"
        )
    else:
        print("✓ RMM memory pooling disabled")

    print_memory_usage("Initial GPU memory: ")

    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    ## Scaffold_GS##
    gaussians = GaussianModel(hyper,dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    ################
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    adaptive_cam = []
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer,adaptive_cam)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer,adaptive_cam)
    print("adaptive stage starts, adaptive_cam length: ",len(adaptive_cam))
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "adaptive", tb_writer, opt.adaptive_iterations,timer,adaptive_cam)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        scene.gaussians.eval()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_gaussian_size, iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_gaussian_size, iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
        scene.gaussians.train()
# Moved to device_utils.py as setup_reproducible_training
if __name__ == "__main__":
    
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_reproducible_training(DEFAULT_RANDOM_SEED)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default=DEFAULT_GUI_IP)
    parser.add_argument('--port', type=int, default=DEFAULT_GUI_PORT)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=TEST_ITERATIONS)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=CHECKPOINT_SAVE_INTERVALS)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        from mmengine import Config
        from utils.params_utils import merge_hparams
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
