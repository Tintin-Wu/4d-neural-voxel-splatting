#!/usr/bin/env python3
"""
Simple PLY fixer that adds normals
Usage: python simple_ply_fixer.py <input_ply> <output_ply>
"""

import numpy as np
import sys
import open3d as o3d

def fix_ply_file(input_path, output_path):
    """Fix PLY file by adding normals using Open3D."""
    
    print(f"Reading PLY file: {input_path}")
    
    # Read with Open3D (more robust)
    pcd = o3d.io.read_point_cloud(input_path)
    
    if len(pcd.points) == 0:
        print("Error: No points found in PLY file")
        return False
    
    print(f"Loaded {len(pcd.points)} points")
    
    # Ensure we have colors (if not, add default ones)
    if len(pcd.colors) == 0:
        print("No colors found, adding default gray colors")
        colors = np.ones((len(pcd.points), 3)) * 0.7
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Zero normals are fine — they are not used downstream.
    print("Adding zero normals")
    normals = np.zeros((len(pcd.points), 3))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Save the fixed PLY file
    success = o3d.io.write_point_cloud(output_path, pcd)
    
    if success:
        print(f"✓ Successfully saved fixed PLY file: {output_path}")
        return True
    else:
        print(f"✗ Failed to save PLY file: {output_path}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_ply_fixer.py <input_ply> <output_ply>")
        print("Example: python simple_ply_fixer.py ../data/Ub4d/Cactus_1-13/points3D.ply ../data/Ub4d/Cactus_1-13/points3D_fixed.ply")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        if fix_ply_file(input_path, output_path):
            print("✓ PLY file fixed successfully!")
        else:
            print("✗ Failed to fix PLY file")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()