_base_ = './dnerf_default.py'
ModelParams = dict(
    voxel_size = 0,
    update_init_factor=4,
    feat_dim = 32,
    n_offsets = 10
)
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 100]
    }
)