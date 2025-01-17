# utils.py

import numpy as np
from noise import pnoise2


def scale_array(arr, min_z, max_z):
    # Find the current min and max of the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Scale the array values to the range [min_z, max_z]
    scaled_arr = (arr - arr_min) / (arr_max - arr_min) * (max_z - min_z) + min_z
    
    return scaled_arr


def project_points_to_image(points_3d, camera, image_width, image_height):
    """
    Projects 3D points onto the image plane using the camera parameters.

    Parameters:
    - points_3d: Nx3 numpy array of 3D points in camera coordinates.
    - camera: PyVista camera object.
    - image_width: Width of the image.
    - image_height: Height of the image.

    Returns:
    - points_2d: Nx2 numpy array of projected 2D points in image coordinates.
    """
    if camera.parallel_projection:
        # Orthographic projection
        scale = 1.0 / camera.parallel_scale
        x = points_3d[:, 0] * scale
        y = points_3d[:, 1] * scale

        # Map to pixel coordinates
        u = (x + 1) * (image_width / 2)
        v = (1 - y) * (image_height / 2)
    else:
        # Perspective projection
        fov = camera.GetViewAngle()
        near = camera.GetClippingRange()[0]
        aspect_ratio = image_width / image_height
        f = 1 / np.tan(np.deg2rad(fov) / 2)

        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        u = (f * x / z) * (image_width / 2) + (image_width / 2)
        v = (f * y / z) * (image_height / 2) + (image_height / 2)

    points_2d = np.vstack((u, v)).T
    return points_2d


def tile_texture(mesh, tiles_x=1, tiles_y=1):
    """Tiles the texture coordinates of a mesh."""
    # Get the texture coordinates
    tex_coords = mesh.active_texture_coordinates.copy()
    # Scale the texture coordinates to tile the texture
    tex_coords[:, 0] *= tiles_x
    tex_coords[:, 1] *= tiles_y
    # Update the mesh texture coordinates
    mesh.active_texture_coordinates = tex_coords

def calculate_perlin_noise(points, scale, octaves, persistence, lacunarity, noise_scale):
    """Generates Perlin noise for given points."""
    x_coords = points[:, 0] * scale
    y_coords = points[:, 1] * scale
    noise_values = np.array([
        pnoise2(x, y, octaves=octaves, persistence=persistence,
                lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=0)
        for x, y in zip(x_coords, y_coords)
    ])
    # Normalize noise values to the desired amplitude
    noise_values = noise_values * noise_scale
    return noise_values

def calculate_intersection_volume_percentage(points_obj1, points_obj2, N_voxels=100):
    """Calculates the percentage of volume overlap between two point clouds."""
    # Combine points to get the bounding box
    all_points = np.vstack((points_obj1, points_obj2))
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)

    # Compute voxel size
    voxel_size = (max_coords - min_coords) / N_voxels

    # Handle zero voxel size in any dimension
    voxel_size[voxel_size == 0] = 1e-6

    # Map points to voxel indices
    voxel_indices_obj1 = np.floor((points_obj1 - min_coords) / voxel_size).astype(int)
    voxel_indices_obj2 = np.floor((points_obj2 - min_coords) / voxel_size).astype(int)

    # Ensure indices are within the grid
    voxel_indices_obj1 = np.clip(voxel_indices_obj1, 0, N_voxels - 1)
    voxel_indices_obj2 = np.clip(voxel_indices_obj2, 0, N_voxels - 1)

    # Compute unique integer indices for voxels
    def compute_voxel_indices(ixyz, N_voxels):
        return ixyz[:, 0] + ixyz[:, 1] * N_voxels + ixyz[:, 2] * N_voxels * N_voxels

    voxel_indices_obj1_int = compute_voxel_indices(voxel_indices_obj1, N_voxels).astype(np.int64)
    voxel_indices_obj2_int = compute_voxel_indices(voxel_indices_obj2, N_voxels).astype(np.int64)

    # Get unique voxels occupied by each object
    voxels_obj1 = np.unique(voxel_indices_obj1_int)
    voxels_obj2 = np.unique(voxel_indices_obj2_int)

    # Compute intersection voxels
    voxels_intersection = np.intersect1d(voxels_obj1, voxels_obj2)

    # Compute volumes
    voxel_volume = np.prod(voxel_size)
    volume_obj1 = len(voxels_obj1) * voxel_volume
    volume_obj2 = len(voxels_obj2) * voxel_volume
    volume_intersection = len(voxels_intersection) * voxel_volume

    # Compute percentage of volume in intersection
    percent_obj1 = (volume_intersection / volume_obj1) * 100 if volume_obj1 > 0 else 0
    percent_obj2 = (volume_intersection / volume_obj2) * 100 if volume_obj2 > 0 else 0

    return percent_obj1, percent_obj2

# Custom JSON encoder for NumPy data types
import json
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return super(NumpyEncoder, self).default(obj)
