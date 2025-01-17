# mask_generation.py

import pyvista as pv
import numpy as np
import cv2
import json
from pycocotools import mask as maskUtils
from utils import NumpyEncoder
from config import (
    CATEGORY_ID, CATEGORY_NAME, SUPERCATEGORY,
    SCENE_CAP_MASK_PATH, SCENE_ANNOTATIONS_PATH, WINDOW_SIZE
)

def generate_masks_and_annotations(cap_meshes, plotter, mask_output_path, annotations_output_path):
    """Generates masks and COCO annotations for cap meshes."""
    # Get the mask image dimensions from the cropped area
    mask_height, mask_width = WINDOW_SIZE

    # Initialize the mask image
    cap_mask_cropped = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Loop over each cap mesh to generate individual masks
    annotations = []
    num_caps = len(cap_meshes)
    cap_meshes = sorted(cap_meshes, key=lambda x: x.bounds[-1])  # Sort by zmax ascending
    for cap_index, cap_mesh in enumerate(cap_meshes):
        label = cap_index + 1  # Labels start from 1

        # Create a new plotter for each cap
        cap_plotter = pv.Plotter(off_screen=True, window_size=plotter.window_size)
        cap_plotter.camera = plotter.camera
        cap_plotter.set_background('black')
        cap_plotter.add_mesh(
            cap_mesh,
            color='white',
            show_edges=False,
            reset_camera=False,
            pickable=False,
            render=False,
            lighting=False,
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
        )
        cap_plotter.show_axes = False
        cap_plotter.remove_bounds_axes()
        cap_plotter.disable_anti_aliasing()
        cap_plotter.render()

        # Get the rendered image
        cap_image = cap_plotter.screenshot(transparent_background=False, return_img=True)

        # Optionally crop the image

        # Convert to grayscale
        cap_image_gray = cv2.cvtColor(cap_image, cv2.COLOR_RGB2GRAY)

        # Threshold to create binary mask
        _, cap_binary = cv2.threshold(cap_image_gray, 1, 1, cv2.THRESH_BINARY)

        # Assign label to mask
        cap_mask_cropped[cap_binary == 1] = label

        # Encode the mask using COCO format
        encoded_mask = maskUtils.encode(np.asfortranarray(cap_binary))

        # Decode counts for JSON serialization
        encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')

        # Compute area
        area = maskUtils.area(encoded_mask)

        # Compute bounding box
        bbox = maskUtils.toBbox(encoded_mask).tolist()

        # Create annotation
        annotation = {
            'id': label,
            'image_id': 1,  # Update image_id if needed
            'category_id': CATEGORY_ID,
            'segmentation': encoded_mask,
            'area': float(area),
            'bbox': bbox,
            'iscrowd': 0
        }
        annotations.append(annotation)

        # Close the plotter to free resources
        cap_plotter.close()

    # Save the cap mask image
    cv2.imwrite(mask_output_path, cap_mask_cropped)

    # Define categories
    categories = [
        {
            'id': CATEGORY_ID,
            'name': CATEGORY_NAME,
            'supercategory': SUPERCATEGORY
        }
    ]

    # Define images
    images = [
        {
            'id': 1,
            'width': cap_mask_cropped.shape[1],
            'height': cap_mask_cropped.shape[0],
            'file_name': SCENE_CAP_MASK_PATH
        }
    ]

    # Assemble COCO data
    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # Save the COCO data as a JSON file
    with open(annotations_output_path, 'w') as f:
        json.dump(coco_data, f, indent=4, cls=NumpyEncoder)
