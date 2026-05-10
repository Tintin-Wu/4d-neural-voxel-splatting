#!/usr/bin/env python3
"""
Convert COLMAP points3D.bin to PLY format
Usage: python colmap_to_ply.py <path_to_points3D.bin> <output.ply>
"""

import sys
import struct
import numpy as np

def read_next_bytes(fid, num_bytes, format_char_sequence):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack('<' + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """Read points3D.bin file and return points and colors."""
    points = []
    colors = []
    
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(f"Reading {num_points} points...")
        
        for i in range(num_points):
            # Read point properties: id, xyz, rgb, error
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            
            # Read track length and skip track data
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length, format_char_sequence="ii"*track_length)
            
            points.append(xyz)
            colors.append(rgb)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_points} points")
    
    return np.array(points), np.array(colors)

def write_ply(filename, points, colors):
    """Write points and colors to PLY file."""
    num_points = len(points)
    
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(num_points):
            f.write(f"{points[i][0]:.6f} {points[i][1]:.6f} {points[i][2]:.6f} "
                   f"{int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n")
    
    print(f"Saved {num_points} points to {filename}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python colmap_to_ply.py <path_to_points3D.bin> <output.ply>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Converting {input_file} to {output_file}")
    
    # Read COLMAP points
    points, colors = read_points3D_binary(input_file)
    
    # Write PLY file
    write_ply(output_file, points, colors)
    
    print("Conversion completed!")

if __name__ == "__main__":
    main()