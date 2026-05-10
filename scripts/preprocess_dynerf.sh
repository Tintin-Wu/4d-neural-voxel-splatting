python scripts/preprocess_dynerf.py --datadir ./data/dynerf/coffee_martini 
python scripts/preprocess_dynerf.py --datadir ./data/dynerf/cook_spinach 
python scripts/preprocess_dynerf.py --datadir ./data/dynerf/cut_roasted_beef 
python scripts/preprocess_dynerf.py --datadir ./data/dynerf/flame_salmon_1 
python scripts/preprocess_dynerf.py --datadir ./data/dynerf/flame_steak 
python scripts/preprocess_dynerf.py --datadir ./data/dynerf/sear_steak 

# wait

# bash colmap.sh ../data/dynerf/coffee_martini llff
# bash colmap.sh ../data/dynerf/cook_spinach llff
# bash colmap.sh ../data/dynerf/cut_roasted_beef llff
# bash colmap.sh ../data/dynerf/flame_salmon_1 llff
# bash colmap.sh ../data/dynerf/flame_steak llff
# bash colmap.sh ../data/dynerf/sear_steak llff

# wait
# python scripts/downsample_point.py ../data/dynerf/coffee_martini/colmap/dense/workspace/fused.ply ../data/dynerf/coffee_martini/points3D_downsample2.ply
# python scripts/downsample_point.py ../data/dynerf/cook_spinach/colmap/dense/workspace/fused.ply ../data/dynerf/cook_spinach/points3D_downsample2.ply
# python scripts/downsample_point.py ../data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply ../data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# python scripts/downsample_point.py ../data/dynerf/flame_salmon_1/colmap/dense/workspace/fused.ply ../data/dynerf/flame_salmon_1/points3D_downsample2.ply
# python scripts/downsample_point.py ../data/dynerf/flame_steak/colmap/dense/workspace/fused.ply ../data/dynerf/flame_steak/points3D_downsample2.ply
# python scripts/downsample_point.py ../data/dynerf/sear_steak/colmap/dense/workspace/fused.ply ../data/dynerf/sear_steak/points3D_downsample2.ply
