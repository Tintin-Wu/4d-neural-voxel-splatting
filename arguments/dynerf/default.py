ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=True,
    static_mlp=False
)
# ModelParams = dict(
#     # appearance_dim =32,
#     voxel_size = 0.005,
# )
OptimizationParams = dict(
    dataloader=True,  # Disable dataloader to reduce memory usage
    iterations = 14000,
    batch_size=2,  # Reduced from 2 to 1 for memory efficiency
    coarse_iterations =3000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    position_lr_max_steps =14000,
    offset_lr_max_steps = 14000,
    mlp_opacity_lr_max_steps = 14000,
    mlp_cov_lr_max_steps = 14000,
    mlp_color_lr_max_steps = 14000,
    mlp_featurebank_lr_max_steps = 14000,
    appearance_lr_max_steps = 14000,
    # Memory optimization settings
    enable_memory_optimization = True,
    gradient_accumulation_steps = 2,  # Compensate for smaller batch size
    empty_cache_frequency = 100,  # Clear cache every N iterations

    # Adaptive training settings (disabled for large datasets like DyNeRF)
    enable_quality_weighting = False,
    enable_adaptive_lr = False,

    # Compact adaptive training (minimal memory overhead)
    enable_compact_adaptive = True,  # Can be enabled even for large datasets
    compact_adaptive_warmup = 1000,

    # Lightweight crude view detection (alternative to EMA PSNR)
    lightweight_quality_method = "hybrid",  # "simple", "periodic", "gradient", "hybrid"
    quality_check_interval = 500,  # For periodic method
    crude_view_threshold = 25.0
)