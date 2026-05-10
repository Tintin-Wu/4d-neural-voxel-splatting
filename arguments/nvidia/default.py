# Fixed configuration to prevent green output in fine stage
ModelHiddenParams = dict()
ModelHiddenParams['net_width'] = 64
ModelHiddenParams['timebase_pe'] = 4
ModelHiddenParams['defor_depth'] = 1  # REDUCED: Prevents overfitting in fine stage
ModelHiddenParams['posebase_pe'] = 10
ModelHiddenParams['scale_rotation_pe'] = 2
ModelHiddenParams['opacity_pe'] = 2
ModelHiddenParams['timenet_width'] = 64
ModelHiddenParams['timenet_output'] = 32
ModelHiddenParams['bounds'] = 1.6

# CRITICAL: Higher regularization to prevent artifacts
ModelHiddenParams['plane_tv_weight'] = 0.001  # Increased
ModelHiddenParams['time_smoothness_weight'] = 0.1  # Increased
ModelHiddenParams['l1_time_planes'] = 0.001  # Increased

# FIXED: Proper temporal grid (adjust 20 to your actual frame count / 2)
ModelHiddenParams['kplanes_config'] = {
    'grid_dimensions': 2,
    'input_coordinate_dim': 4,
    'output_coordinate_dim': 32,
    'resolution': [64, 64, 64, 20]  # Set to num_frames // 2
}

ModelHiddenParams['multires'] = [1, 2, 4, 8]

# CRITICAL: Disable problematic deformations
ModelHiddenParams['no_dx'] = False  # Keep position deformation
ModelHiddenParams['no_grid'] = False  # Keep temporal grid
ModelHiddenParams['no_ds'] = False  # Keep scale deformation
ModelHiddenParams['no_dr'] = False  # Keep rotation deformation
ModelHiddenParams['no_do'] = True   # DISABLE opacity deformation (fixes green output)
ModelHiddenParams['no_dshs'] = True # DISABLE SH deformation (fixes green output)

OptimizationParams = dict()
OptimizationParams['dataloader'] = True
OptimizationParams['iterations'] = 20000

# REDUCED learning rates for stability
OptimizationParams['position_lr_init'] = 0.0001
OptimizationParams['position_lr_final'] = 0.000001
OptimizationParams['position_lr_delay_mult'] = 0.01
OptimizationParams['position_lr_max_steps'] = 20000

OptimizationParams['deformation_lr_init'] = 0.0001
OptimizationParams['deformation_lr_final'] = 0.00001
OptimizationParams['deformation_lr_delay_mult'] = 0.01

OptimizationParams['grid_lr_init'] = 0.001
OptimizationParams['grid_lr_final'] = 0.0001

OptimizationParams['feature_lr'] = 0.0025
OptimizationParams['opacity_lr'] = 0.05
OptimizationParams['scaling_lr'] = 0.005
OptimizationParams['rotation_lr'] = 0.001

OptimizationParams['percent_dense'] = 0.01
OptimizationParams['lambda_dssim'] = 0.2

# ADJUSTED densification to prevent over-densification
OptimizationParams['densification_interval'] = 200
OptimizationParams['opacity_reset_interval'] = 2000
OptimizationParams['densify_from_iter'] = 500
OptimizationParams['densify_until_iter'] = 12000
OptimizationParams['densify_grad_threshold'] = 0.0003

OptimizationParams['random_background'] = False