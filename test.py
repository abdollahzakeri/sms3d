import os
import json
import numpy as np
import cv2
import pyvista as pv

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
    # Check if parallel projection is enabled
    if camera.parallel_projection:
        # For parallel projection, projection is orthographic
        # Compute the scaling factors
        scale_x = 1.0 / camera.parallel_scale
        scale_y = 1.0 / camera.parallel_scale

        # Project the 3D points
        x = points_3d[:, 0] * scale_x
        y = points_3d[:, 1] * scale_y

        # Map to pixel coordinates
        # Assuming the principal point is at the center of the image
        u = (x + 1) * (image_width / 2)
        v = (1 - y) * (image_height / 2)
    else:
        # Perspective projection
        # Get camera intrinsic parameters
        fx = camera.GetEffectiveFocalLength()  # Focal length in pixels
        fy = fx  # Assuming square pixels
        cx = image_width / 2
        cy = image_height / 2

        # Project the 3D points
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        u = (fx * x / z) + cx
        v = (fy * y / z) + cy

    points_2d = np.vstack([u, v]).T
    return points_2d

def overlay_points_on_image(image_path, annotations_path, camera, output_path):
    # Read the RGB image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Read the annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Extract 3D centers
    centers_3d = []
    for obj in annotations:
        center = obj['center']
        centers_3d.append(center)
    centers_3d = np.array(centers_3d)

    # Project the 3D centers onto the image plane
    points_2d = project_points_to_image(centers_3d, camera, image_width, image_height)
    points_2d = points_2d.astype(int)

    # Overlay red dots on the image
    for point in points_2d:
        x, y = point
        cv2.circle(image, (x, y), radius=50, color=(0, 0, 255), thickness=-1)

    # Save the resulting image
    cv2.imwrite(output_path, image)
    print(f"Saved image with overlaid points to {output_path}")

def main():
    # Paths
    scene_dir = 'scenes/1'  # Replace with your scene directory
    image_path = os.path.join(scene_dir, 'scene_rgb.png')
    annotations_path = os.path.join(scene_dir, 'cap_bounding_boxes_3d_camera.json')
    output_path = os.path.join(scene_dir, 'scene_with_centers.png')

    # Create a PyVista plotter to access the camera parameters
    # Note: Ensure that the plotter settings match those used during rendering
    plotter = pv.Plotter(window_size=(1920, 1080), off_screen=True)
    plotter.view_xy()
    plotter.camera.zoom(1.0)
    plotter.enable_parallel_projection()
    plotter.render()

    camera = plotter.camera

    # Overlay points on image
    overlay_points_on_image(image_path, annotations_path, camera, output_path)

if __name__ == '__main__':
    main()
