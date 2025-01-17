# main.py

import os

os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_OSMESA'] = 'true'

import argparse
import pyvista as pv
import numpy as np
import cv2
import random
import json
from tqdm import tqdm
from multiprocessing import Pool
from config import (
    WINDOW_SIZE, BACKGROUND_COLOR,
    X_RANGE, Y_RANGE, MAX_ATTEMPTS, NEAREST_NEIGHBORS,
    RANDOM_SEED, GROUND_I_SIZE
)
from data_loader import load_soil_textures, load_mushroom_models, select_random_soil_texture
from scene_generation import create_ground_plane, create_random_mushroom
from utils import calculate_intersection_volume_percentage, project_points_to_image, scale_array
from mask_generation import generate_masks_and_annotations
from scipy.spatial.transform import Rotation as R


def generate_scene_wrapper(args):
    generate_scene(*args)
    return None

def generate_scene(scene_num, gpu_id):
    
    # Load textures and models
    soil_textures = load_soil_textures()
    mushroom_models = load_mushroom_models()

    # Select a random soil texture
    soil_texture = select_random_soil_texture(soil_textures)

    # Create ground plane
    ground, ground_interpolator = create_ground_plane()

    # Initialize lists
    mushrooms = []
    textures = []
    cap_meshes = []
    cap_boxes = []
    existing_mushrooms = []
    rotation_matrices = []

    # Mushroom placement loop
    num_mushrooms = random.randint(1, 100)  # Adjust as needed
    for _ in range(num_mushrooms):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            # Random position within the given range
            x_pos = random.uniform(*X_RANGE)
            y_pos = random.uniform(*Y_RANGE)

            # Create a mushroom at this position
            mushroom_data = create_random_mushroom(mushroom_models, (x_pos, y_pos), ground_interpolator)
            mushroom, texture, cap_mesh, cap_box, x_center, y_center, effective_radius, rotation_matrix = mushroom_data

            # Check for overlaps
            overlap = False
            existing_mushrooms.sort(key=lambda x: np.linalg.norm(np.array([x[1], x[2]]) - np.array([x_center, y_center])))
            for ex_mushroom, ex_x, ex_y, ex_radius in existing_mushrooms[:NEAREST_NEIGHBORS]:
                percent_obj1, percent_obj2 = calculate_intersection_volume_percentage(ex_mushroom.points, mushroom.points)
                if percent_obj1 > 0 or percent_obj2 > 0:
                    overlap = True
                    break

            if not overlap:
                mushrooms.append(mushroom)
                textures.append(texture)
                cap_meshes.append(cap_mesh)
                cap_boxes.append(cap_box)
                rotation_matrices.append(rotation_matrix)
                existing_mushrooms.append((mushroom, x_center, y_center, effective_radius))
                break
        else:
            pass

    # Create a scene-specific directory
    scenes_dir = 'scenes'
    scene_dir = os.path.join(scenes_dir, str(scene_num))
    os.makedirs(scene_dir, exist_ok=True)

    #print(f"Generating scene {scene_num} on GPU {gpu_id} in {scene_dir}")

    # Define output paths
    SCENE_RGB_PATH = os.path.join(scene_dir, 'scene_rgb.png')
    SCENE_DEPTH_PATH = os.path.join(scene_dir, 'scene_depth.npy')
    SCENE_CROPPED_RGB_PATH = os.path.join(scene_dir, 'scene_rgb_cropped.png')
    SCENE_CROPPED_DEPTH_PATH = os.path.join(scene_dir, 'scene_depth_cropped.npy')
    SCENE_PLY_PATH = os.path.join(scene_dir, 'scene.ply')
    CAP_BBOXES_3D_WORLD_PATH = os.path.join(scene_dir, 'cap_bounding_boxes_3d_world.json')
    CAP_BBOXES_3D_CAMERA_PATH = os.path.join(scene_dir, 'cap_bounding_boxes_3d_camera.json')
    CAP_BBOXES_2D_PATH = os.path.join(scene_dir, 'cap_bounding_boxes_2d.json')
    SCENE_CAP_MASK_PATH = os.path.join(scene_dir, 'scene_cap_mask.png')
    SCENE_ANNOTATIONS_PATH = os.path.join(scene_dir, 'scene_annotations.json')
    CAP_BBOXES_PIXELS_PATH = os.path.join(scene_dir, 'cap_bounding_boxes_pixels.json')

    # Create a PyVista Plotter
    plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=True)
    plotter.add_mesh(ground, texture=soil_texture, show_edges=False)

    # Add mushrooms to the plotter
    for mushroom, texture in zip(mushrooms, textures):
        plotter.add_mesh(mushroom, texture=texture, show_edges=False)

    plotter.view_xy()
    plotter.camera.zoom(1)

    # Enable parallel projection (useful for orthographic views like top-down)
    plotter.enable_parallel_projection()
    plotter.show_axes = True
    plotter.remove_bounds_axes()
    plotter.disable_anti_aliasing()

    plotter.set_background(BACKGROUND_COLOR)

    # Reset clipping range to ensure the scene is fully rendered
    plotter.camera.reset_clipping_range()

    # Render the plotter to apply the initial changes
    plotter.render()

    # Get camera parameters after setting up the view
    camera_position = np.array(plotter.camera.position)  # Example: (0.0, 0.0, 138.81557643990826)
    camera_focal_point = np.array(plotter.camera.focal_point)  # Example: (0.0, 0.0, 0.0)
    camera_up = np.array(plotter.camera.up)  # Example: (0.0, 1.0, 0.0)
    camera_fov = plotter.camera.view_angle  # Get the camera's field of view in degrees

    # Given: size of the square (L) and camera height (h)
    L = 50  # Size of the square (adjust as needed)
    h = np.linalg.norm(camera_position - camera_focal_point)  # Calculate height (distance to focal point)

    # Convert FOV from degrees to radians and calculate the initial view size
    fov_rad = np.radians(camera_fov)
    initial_view_size = 2 * h * np.tan(fov_rad / 2)  # Initial view size considering the FOV

    # Calculate zoom factor to fit the square into the view
    zoom_factor = initial_view_size / L

    # Apply the zoom factor
    plotter.camera.zoom(zoom_factor * 0.966)

    # Render the plotter again after modifying the zoom
    plotter.render()

    # Compute camera coordinate system axes
    z_axis = (camera_focal_point - camera_position)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(camera_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix from world to camera coordinates
    R_world_to_camera = np.vstack([x_axis, y_axis, z_axis]).T

    # Translation vector
    T_world_to_camera = -R_world_to_camera @ camera_position

    # Transformation matrix
    T_world_to_camera_matrix = np.eye(4)
    T_world_to_camera_matrix[:3, :3] = R_world_to_camera
    T_world_to_camera_matrix[:3, 3] = T_world_to_camera

    # Render the scene and save the RGB image
    rgb_image = plotter.screenshot(SCENE_RGB_PATH)
    height, width, _ = rgb_image.shape

    


    # **Crop the RGB and Depth Images to the Ground Plane Bounds**

    # Initialize the VTK coordinate system
    coordinate = pv._vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()

    # Get ground plane bounds
    xmin, xmax, ymin, ymax, zmin, zmax = ground.bounds
    z_mean = np.mean(ground.points[:, 2])

    # Define the corner points of the ground plane
    ground_corners_world = np.array([
        [xmin, ymin, z_mean],
        [xmax, ymin, z_mean],
        [xmax, ymax, z_mean],
        [xmin, ymax, z_mean]
    ])

    # Transform ground corners to camera coordinates
    ground_corners_world_homogeneous = np.hstack((ground_corners_world, np.ones((4, 1))))
    ground_corners_camera_homogeneous = (T_world_to_camera_matrix @ ground_corners_world_homogeneous.T).T
    ground_corners_camera = ground_corners_camera_homogeneous[:, :3]

    # Project ground corners onto image plane
    ground_corners_2d = project_points_to_image(ground_corners_camera, plotter.camera, width, height)

    # Get the min and max x and y to define the crop rectangle
    u_coords = ground_corners_2d[:, 0]
    v_coords = ground_corners_2d[:, 1]
    x_min, x_max = int(u_coords.min()), int(u_coords.max())
    y_min, y_max = int(v_coords.min()), int(v_coords.max())

    # Ensure crop coordinates are within image bounds
    x_min = max(0, x_min)
    x_max = min(width, x_max)
    y_min = max(0, y_min)
    y_max = min(height, y_max)



    height, width, _ = rgb_image.shape

    # Project the centers of the bounding boxes
    centers_camera = []
    depths = []

    for rot, cap_box in zip(rotation_matrices, cap_boxes):
        # Get center in world coordinates
        center_world = cap_box.center

        # Transform center to camera coordinates
        center_world_homogeneous = np.append(center_world, 1)
        center_camera_homogeneous = T_world_to_camera_matrix @ center_world_homogeneous
        center_camera = center_camera_homogeneous[:3]

        centers_camera.append(center_camera)
        depths.append(center_camera[2])  # Depth value

    # Convert centers to NumPy array
    centers_camera = np.array(centers_camera)

    # Project centers onto image plane to get pixel coordinates
    centers_2d = project_points_to_image(centers_camera, plotter.camera, width, height)

    # Prepare the data for the new JSON file
    cap_bounding_boxes_pixels = []
    
    C = np.array(   [[1,  0,  0],
                        [0, -1,  0],
                        [0,  0,  1]])
        

    for rot, cap_box in zip(rotation_matrices, cap_boxes):
        # Get center and dimensions in world coordinates
        center_world = np.array(cap_box.center) / GROUND_I_SIZE * WINDOW_SIZE[0]
        center_world[0] += WINDOW_SIZE[0] / 2
        center_world[1] = WINDOW_SIZE[0] / 2 - center_world[1]
        center_world[2] = center_world[2] * 1.2

        dimensions = np.array([
            cap_box.bounds[1] - cap_box.bounds[0],  # length (x)
            cap_box.bounds[3] - cap_box.bounds[2],  # width (y)
            cap_box.bounds[5] - cap_box.bounds[4]   # height (z)
        ])

        dimensions = dimensions / GROUND_I_SIZE * WINDOW_SIZE[0]

        
        #rotation_matrix[1,:] = -rotation_matrix[1,:]
        rot = C @ np.array(rot) @ C

        rotation_world = R.from_matrix(rot)

        # corners = np.array(cap_box.points)
        # for i in range(len(corners)):
        #     corners[i] = np.array(corners[i]) / GROUND_I_SIZE * WINDOW_SIZE[0]
        #     corners[i][0] += WINDOW_SIZE[0] / 2
        #     corners[i][1] = WINDOW_SIZE[0] / 2 - corners[i][1]

        # Save pixel coordinates annotations
        cap_bounding_boxes_pixels.append({
            'center': center_world.tolist(),
            'dimensions': dimensions.tolist(),
            'rotation': rotation_world.as_matrix().tolist(),
            # 'corners':  corners.tolist(),
            'class': 'mushroom_cap'
        })


    with open(CAP_BBOXES_PIXELS_PATH, 'w') as f:
        json.dump(cap_bounding_boxes_pixels, f, indent=4)

    # Save the scene mesh
    scene_mesh = ground.copy()
    for mushroom in mushrooms:
        scene_mesh = scene_mesh + mushroom

    scene_mesh.save(SCENE_PLY_PATH)


    # Get the depth map from the current view
    depth_image = plotter.get_image_depth()

    all_points_world = scene_mesh.points  # Nx3 array of points in world coordinates

    # Transform points to camera coordinates
    ones = np.ones((all_points_world.shape[0], 1))
    all_points_world_homogeneous = np.hstack([all_points_world, ones])
    all_points_camera_homogeneous = (T_world_to_camera_matrix @ all_points_world_homogeneous.T).T
    all_points_camera = all_points_camera_homogeneous[:, :3]

    # Get min and max Z-values from the scene
    min_z = np.min(all_points_camera[:, 2])
    max_z = np.max(all_points_camera[:, 2])

    # Normalize z_cam between min_z and max_z
    scaled_depth_image = scale_array(depth_image,min_z,max_z)
    scaled_depth_image -= np.min(scaled_depth_image)

    scaled_depth_image -= np.min(scaled_depth_image)
    scaled_depth_image = np.max(scaled_depth_image) - scaled_depth_image
    scaled_depth_image = scaled_depth_image / GROUND_I_SIZE * WINDOW_SIZE[0]
    np.save(SCENE_DEPTH_PATH, scaled_depth_image)




    # Save 3D bounding box parameters in world coordinates
    cap_bounding_boxes_3d_world = []

    for rot, cap_box in zip(rotation_matrices, cap_boxes):
        # Get center and dimensions in world coordinates
        center_world = cap_box.center
        dimensions = np.array([
            cap_box.bounds[1] - cap_box.bounds[0],  # length (x)
            cap_box.bounds[3] - cap_box.bounds[2],  # width (y)
            cap_box.bounds[5] - cap_box.bounds[4]   # height (z)
        ])

        # Convert orientation matrix to quaternion
        rotation_world = R.from_matrix(rot)

        # Save world coordinates annotations
        cap_bounding_boxes_3d_world.append({
            'center': center_world,
            'dimensions': dimensions.tolist(),
            'rotation': rotation_world.as_matrix().tolist(),
            'class': 'mushroom_cap'
        })

    # Save world coordinates bounding boxes
    with open(CAP_BBOXES_3D_WORLD_PATH, 'w') as f:
        json.dump(cap_bounding_boxes_3d_world, f, indent=4)


    # Generate masks and annotations
    generate_masks_and_annotations(cap_meshes, plotter,
                                   SCENE_CAP_MASK_PATH, SCENE_ANNOTATIONS_PATH)

    #print(f"Scene {scene_num} generation completed.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic scenes.')
    parser.add_argument('--start_scene', type=int, default=1, help='Starting scene number.')
    parser.add_argument('--num_scenes', type=int, default=1, help='Number of scenes to generate.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
    args = parser.parse_args()

    # Create a list of scene numbers
    scene_numbers = list(range(args.start_scene, args.start_scene + args.num_scenes))

    # Assign scenes to GPUs in a round-robin fashion
    tasks = [(scene_num, gpu_id % args.num_gpus) for gpu_id, scene_num in enumerate(scene_numbers)]

    # Create a pool of workers equal to the number of GPUs
    with Pool(processes=args.num_gpus) as pool:
        # Use imap_unordered to get an iterator and wrap it with tqdm for progress tracking
        for _ in tqdm(pool.imap_unordered(generate_scene_wrapper, tasks), total=len(tasks), desc='Generating Scenes'):
            pass  # We don't need to do anything here; the loop is just for progress tracking

if __name__ == '__main__':
    main()
