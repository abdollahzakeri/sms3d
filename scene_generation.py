# scene_generation.py

import numpy as np
import pyvista as pv
import random
from scipy.interpolate import LinearNDInterpolator
from config import (
    GROUND_I_SIZE, GROUND_J_SIZE, GROUND_I_RESOLUTION, GROUND_J_RESOLUTION,
    NOISE_SCALE, PERLIN_SCALE, OCTAVES, PERSISTENCE, LACUNARITY,
    GROUND_TEXTURE_TILES_X, GROUND_TEXTURE_TILES_Y,
    ANGLE_X_STDDEV, ANGLE_Y_STDDEV, SCALE_RANGE, STRETCH_RANGE,
    VECTOR_LENGTH, ANGLE_RANGE
)
from utils import tile_texture, calculate_perlin_noise

def create_ground_plane():
    """Creates a ground plane with Perlin noise."""
    ground = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=GROUND_I_SIZE,
        j_size=GROUND_J_SIZE,
        i_resolution=GROUND_I_RESOLUTION,
        j_resolution=GROUND_J_RESOLUTION
    )

    # Get the points of the plane
    points = ground.points.copy()

    # Generate Perlin noise
    noise_values = calculate_perlin_noise(points, PERLIN_SCALE, OCTAVES, PERSISTENCE, LACUNARITY, NOISE_SCALE)

    # Add noise to the Z-coordinate
    points[:, 2] += noise_values

    # Update the plane's points
    ground.points = points

    # Recompute normals for proper lighting
    ground.compute_normals(inplace=True)

    # Create ground interpolator
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    points2d = np.column_stack((x, y))
    ground_interpolator = LinearNDInterpolator(points2d, z)

    # Tile the ground texture if needed
    tile_texture(ground, tiles_x=GROUND_TEXTURE_TILES_X, tiles_y=GROUND_TEXTURE_TILES_Y)

    return ground, ground_interpolator

def create_random_mushroom(models, position, ground_interpolator):
    """Creates a randomly transformed mushroom at a given position."""
    # Randomly select a mushroom model
    base_mesh, texture, base_cap_mesh, cap_bounds = random.choice(models)

    # Clone the base mesh and cap mesh
    mushroom = base_mesh.copy(deep=True)
    cap_mesh = base_cap_mesh.copy(deep=True)

    # Create a cube representing the cap bounding box
    cap_box = pv.Cube(bounds=cap_bounds)

    # Rotate mushroom upright if necessary
    mushroom.rotate_x(90, point=(0, 0, 0), inplace=True)
    cap_mesh.rotate_x(90, point=(0, 0, 0), inplace=True)
    cap_box.rotate_x(90, point=(0, 0, 0), inplace=True)

    # Random rotation around x, y, z axes
    angle_x = np.abs(random.gauss(0, ANGLE_X_STDDEV))
    angle_y = np.abs(random.gauss(0, ANGLE_Y_STDDEV))
    angle_z = random.uniform(0, 360)

    # angle_x = 0#np.abs(random.gauss(0, ANGLE_X_STDDEV))
    # angle_y = 60#np.abs(random.gauss(0, ANGLE_Y_STDDEV))
    # angle_z = 0#random.uniform(0, 360)

    # Convert angles to radians
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)

    # Define rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
                    [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])

    R_y = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                    [0, 1, 0],
                    [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])

    R_z = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
                    [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
                    [0, 0, 1]])

    # Compute combined rotation matrix
    R = R_z @ R_y @ R_x

    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R

    original_bounds = cap_box.bounds
    original_width = original_bounds[1] - original_bounds[0]
    original_height = original_bounds[3] - original_bounds[2]
    original_depth = original_bounds[5] - original_bounds[4]

    # Apply rotations to the mushroom and cap meshes
    mushroom.transform(R_homogeneous, inplace=True)
    cap_mesh.transform(R_homogeneous, inplace=True)
    cap_box.transform(R_homogeneous, inplace=True)

    # Get the new bounds of the cap_box after rotation and scaling
    rotated_bounds = cap_box.bounds
    rotated_width = rotated_bounds[1] - rotated_bounds[0]
    rotated_height = rotated_bounds[3] - rotated_bounds[2]
    rotated_depth = rotated_bounds[5] - rotated_bounds[4]

    # Calculate the scaling factors due to rotation
    scale_width = original_width / rotated_width
    scale_height = original_height / rotated_height
    scale_depth = original_depth / rotated_depth

    # Center of the cap_box
    center = np.array(cap_box.center)

    # Apply scaling to cap_box to shrink it back to original dimensions
    cap_box.scale([scale_width, scale_height, scale_depth], inplace= True)
    offset = center - np.array(cap_box.center)
    cap_box.translate(offset, inplace=True)

    # Also compute the transformed up vector
    up_vector = np.array([0, 0, 1])  # Original up direction
    up_vector_transformed = R @ up_vector  # Transformed orientation

    # Random scaling and stretching
    stretch_x = random.uniform(*STRETCH_RANGE)
    stretch_y = random.uniform(*STRETCH_RANGE)
    stretch_z = random.uniform(*STRETCH_RANGE)

    scale_factor = random.uniform(*SCALE_RANGE)

    scaling_matrix = np.diag([scale_factor * stretch_x, scale_factor * stretch_y, scale_factor * stretch_z, 1])

    # Apply scaling
    mushroom.transform(scaling_matrix, inplace=True)
    cap_mesh.transform(scaling_matrix, inplace=True)
    cap_box.transform(scaling_matrix, inplace=True)

    # After scaling and rotation, align the mushroom so that its stem base is at z=0
    xmin, xmax, ymin, ymax, zmin, zmax = mushroom.bounds
    # Translate mushroom in z so that zmin is at 0
    mushroom.translate([0, 0, -zmin], inplace=True)
    cap_mesh.translate([0, 0, -zmin], inplace=True)
    cap_box.translate([0, 0, -zmin], inplace=True)

    # Move the mushroom to the specified position at z=0
    mushroom.translate([position[0], position[1], 0], inplace=True)
    cap_mesh.translate([position[0], position[1], 0], inplace=True)
    cap_box.translate([position[0], position[1], 0], inplace=True)

    # Get ground elevation at this position
    z_ground = ground_interpolator((position[0], position[1]))
    if np.isnan(z_ground):
        # If outside ground area, set to 0
        z_ground = 0.0

    # Move mushroom up by z_ground
    mushroom.translate([0, 0, z_ground], inplace=True)
    cap_mesh.translate([0, 0, z_ground], inplace=True)
    cap_box.translate([0, 0, z_ground], inplace=True)

    # Get the center of the mushroom after all transformations
    x_center, y_center, z_center = mushroom.center

    # After all transformations, compute effective radius for collision detection
    xmin, xmax, ymin, ymax, _, _ = mushroom.bounds
    x_extent = xmax - xmin
    y_extent = ymax - ymin
    effective_radius = np.sqrt(x_extent**2 + y_extent**2) / 2

    # Return the mushroom and additional data
    return mushroom, texture, cap_mesh, cap_box, x_center, y_center, effective_radius, R


# def create_random_mushroom(models, position, ground_interpolator):
#     """Creates a randomly transformed mushroom at a given position."""
#     # Randomly select a mushroom model
#     base_mesh, texture, base_cap_mesh, cap_bounds = random.choice(models)

#     # Clone the base mesh and cap mesh
#     mushroom = base_mesh.copy(deep=True)
#     cap_mesh = base_cap_mesh.copy(deep=True)

#     # Create a cube representing the cap bounding box
#     cap_box = pv.Cube(bounds=cap_bounds)

#     # Rotate mushroom upright if necessary
#     mushroom.rotate_x(90, point=(0, 0, 0), inplace=True)
#     cap_mesh.rotate_x(90, point=(0, 0, 0), inplace=True)
#     cap_box.rotate_x(90, point=(0, 0, 0), inplace=True)

#     # Store the original cap_box dimensions before rotation
#     original_bounds = cap_box.bounds
#     original_width = original_bounds[1] - original_bounds[0]
#     original_height = original_bounds[3] - original_bounds[2]
#     original_depth = original_bounds[5] - original_bounds[4]

#     # Random rotation around x, y, z axes
#     angle_x = random.gauss(0, ANGLE_X_STDDEV)
#     angle_y = random.gauss(0, ANGLE_Y_STDDEV)
#     angle_z = random.uniform(0, 360)

#     # Apply rotations to the mushroom and cap meshes
#     mushroom.rotate_z(angle_z, point=(0, 0, 0), inplace=True)
#     mushroom.rotate_y(angle_y, point=(0, 0, 0), inplace=True)
#     mushroom.rotate_x(angle_x, point=(0, 0, 0), inplace=True)

#     cap_mesh.rotate_z(angle_z, point=(0, 0, 0), inplace=True)
#     cap_mesh.rotate_y(angle_y, point=(0, 0, 0), inplace=True)
#     cap_mesh.rotate_x(angle_x, point=(0, 0, 0), inplace=True)

#     cap_box.rotate_z(angle_z, point=(0, 0, 0), inplace=True)
#     cap_box.rotate_y(angle_y, point=(0, 0, 0), inplace=True)
#     cap_box.rotate_x(angle_x, point=(0, 0, 0), inplace=True)

#     # Build the combined rotation matrix based on the rotation angles
#     # Convert angles to radians
#     angle_x_rad = np.radians(angle_x)
#     angle_y_rad = np.radians(angle_y)
#     angle_z_rad = np.radians(angle_z)

#     # Define rotation matrices
#     R_x = np.array([[1, 0, 0],
#                     [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
#                     [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])

#     R_y = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
#                     [0, 1, 0],
#                     [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])

#     R_z = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
#                     [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
#                     [0, 0, 1]])

#     # Compute combined rotation matrix
#     R = R_z @ R_y @ R_x

#     # Random scaling and stretching
#     stretch_x = random.uniform(*STRETCH_RANGE)
#     stretch_y = random.uniform(*STRETCH_RANGE)
#     stretch_z = random.uniform(*STRETCH_RANGE)

#     scale_factor = random.uniform(*SCALE_RANGE)

#     scaling_matrix = np.diag([scale_factor * stretch_x, scale_factor * stretch_y, scale_factor * stretch_z, 1])

#     # Apply scaling
#     mushroom.transform(scaling_matrix, inplace=True)
#     cap_mesh.transform(scaling_matrix, inplace=True)
#     cap_box.transform(scaling_matrix, inplace=True)

#     # After scaling and rotation, align the mushroom so that its stem base is at z=0
#     xmin, xmax, ymin, ymax, zmin, zmax = mushroom.bounds
#     # Translate mushroom in z so that zmin is at 0
#     mushroom.translate([0, 0, -zmin], inplace=True)
#     cap_mesh.translate([0, 0, -zmin], inplace=True)
#     cap_box.translate([0, 0, -zmin], inplace=True)

#     # Move the mushroom to the specified position at z=0
#     mushroom.translate([position[0], position[1], 0], inplace=True)
#     cap_mesh.translate([position[0], position[1], 0], inplace=True)
#     cap_box.translate([position[0], position[1], 0], inplace=True)

#     # Get ground elevation at this position
#     z_ground = ground_interpolator((position[0], position[1]))
#     if np.isnan(z_ground):
#         # If outside ground area, set to 0
#         z_ground = 0.0

#     # Move mushroom up by z_ground
#     mushroom.translate([0, 0, z_ground], inplace=True)
#     cap_mesh.translate([0, 0, z_ground], inplace=True)
#     cap_box.translate([0, 0, z_ground], inplace=True)

#     # **Adjust the cap_box dimensions to eliminate padding caused by rotation**

#     # Get the new bounds of the cap_box after rotation and scaling
#     rotated_bounds = cap_box.bounds
#     rotated_width = rotated_bounds[1] - rotated_bounds[0]
#     rotated_height = rotated_bounds[3] - rotated_bounds[2]
#     rotated_depth = rotated_bounds[5] - rotated_bounds[4]

#     # Calculate the scaling factors due to rotation
#     scale_width = original_width / rotated_width
#     scale_height = original_height / rotated_height
#     scale_depth = original_depth / rotated_depth

#     # Center of the cap_box
#     center = np.array(cap_box.center)

#     # Apply scaling to cap_box to shrink it back to original dimensions
#     cap_box.scale([scale_width, scale_height, scale_depth], inplace= True)
#     offset = center - np.array(cap_box.center)
#     cap_box.translate(offset, inplace=True)

#     # Get the center of the mushroom after all transformations
#     x_center, y_center, z_center = mushroom.center

#     # After all transformations, compute effective radius for collision detection
#     xmin, xmax, ymin, ymax, _, _ = mushroom.bounds
#     x_extent = xmax - xmin
#     y_extent = ymax - ymin
#     effective_radius = np.sqrt(x_extent**2 + y_extent**2) / 2

#     # Return the mushroom and additional data
#     return mushroom, texture, cap_mesh, cap_box, x_center, y_center, effective_radius, R