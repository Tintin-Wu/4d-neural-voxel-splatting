_base_ = './dnerf_default.py'
OptimizationParams = dict(

    coarse_iterations = 3000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 30000,
    pruning_interval = 8000,
    percent_dense = 0.01,
    render_process=False,
    # no_do=False,
    # no_dshs=False
    
    # opacity_reset_interval=30000
)
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 100]
    }
)