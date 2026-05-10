import torch

class AdaptiveVisibility:
    def __init__(self, min_points=50):
        self.min_points = min_points
        
    def ensure_minimum_points(self,voxel_visible_mask,xyz, viewpoint_cam):
        """Ensure we have minimum points for stable training"""
        
        if voxel_visible_mask is None:
            return None
            
        visible_count = voxel_visible_mask.sum()
        
        if visible_count < self.min_points:
            print(f"Expanding visibility: {visible_count} -> {self.min_points} points")
            
            # Strategy 1: Expand the visibility radius
            cam_pos = viewpoint_cam.camera_center
            
            # Find closest points to camera
            distances = torch.norm(xyz - cam_pos.unsqueeze(0), dim=1)
            _, closest_indices = torch.topk(distances, self.min_points, largest=False)
            
            # Create new mask
            new_mask = torch.zeros_like(voxel_visible_mask)
            new_mask[closest_indices] = True
            
            return new_mask
            
        return voxel_visible_mask