ModelHiddenParams = dict(
    net_width=64,
    timebase_pe=4,
    defor_depth=1,
    posebase_pe=10,
    scale_rotation_pe=2,
    opacity_pe=2,
    timenet_width=64,
    timenet_output=32,
    bounds=1.6,
    plane_tv_weight=0.001,
    time_smoothness_weight=0.1,
    l1_time_planes=0.001,
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 25]
    },
    multires=[1, 2, 4, 8],
    no_dx=False,
    no_grid=False,
    no_ds=False,
    no_dr=False,
    no_do=True,
    no_dshs=True,
)

OptimizationParams = dict(
    dataloader=True,
    zerostamp_init=False,
    custom_sampler=None,
    adaptive_strat="psnr",
    iterations=20000,  # Adjusted for stability
    coarse_iterations=3000,
    adaptive_iterations=20000,  # Adjusted for stability
    add_adaptive_cam_from_iter=10000,
    adaptive_per_iter=1,
    ema_grad_offset_init=0.05,
    ema_grad_offset_final=0.02,
    position_lr_init=0.001,  # Reduced for stability
    position_lr_final=0.00001,  # Reduced for stability
    position_lr_delay_mult=0.01,
    position_lr_max_steps=20000,  # Adjusted for stability
)