# config.py

import os

# Directories
PROJECT_ROOT = '.'  # Update if your script is in a different directory
SOIL_DIR = os.path.join(PROJECT_ROOT, 'soil')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'white')

# Noise parameters for ground generation
NOISE_SCALE = 1  # Amplitude of the bumps
PERLIN_SCALE = 0.5  # Scale of the noise
OCTAVES = 6  # Detail of the noise
PERSISTENCE = 0.5  # Amplitude of the noise
LACUNARITY = 2.0  # Frequency of the noise

# Ground plane parameters
GROUND_I_SIZE = 50
GROUND_J_SIZE = 50
GROUND_I_RESOLUTION = 200
GROUND_J_RESOLUTION = 200

# Texture tiling
GROUND_TEXTURE_TILES_X = 1
GROUND_TEXTURE_TILES_Y = 1

# Mushroom generation parameters
ANGLE_RANGE = (-90, 90)
ANGLE_X_STDDEV = 20  # Standard deviation for Gaussian rotation around X-axis
ANGLE_Y_STDDEV = 20  # Standard deviation for Gaussian rotation around Y-axis
SCALE_RANGE = (0.8, 2)
STRETCH_RANGE = (0.8, 1.2)

# Mushroom placement parameters
TOTAL_MUSHROOMS = 75  # You can set a fixed value or keep it random
MAX_ATTEMPTS = 10
X_RANGE = (-18, 18)
Y_RANGE = (-18, 18)
NEAREST_NEIGHBORS = 5  # Number of nearest mushrooms to check for overlap
OVERLAP_THRESHOLD = 0  # Percentage overlap allowed between mushrooms

# Rendering parameters
WINDOW_SIZE = [1024, 1024]
BACKGROUND_COLOR = 'white'
VECTOR_LENGTH = 5  # Length of orientation vector for visualization

# Output files
SCENE_RGB_PATH = 'scene_rgb.png'
SCENE_DEPTH_PATH = 'scene_depth.npy'
SCENE_DEPTH_HEATMAP_PATH = 'scene_depth_heatmap.png'
SCENE_BBOXES_PATH = 'scene_with_bboxes.png'
SCENE_CROPPED_RGB_PATH = 'scene_rgb_cropped.png'
SCENE_CROPPED_BBOXES_PATH = 'scene_with_bboxes_cropped.png'
SCENE_OBJ_PATH = 'scene.obj'
CAP_BBOXES_3D_PATH = 'cap_bounding_boxes_3d.json'
SCENE_CAP_MASK_PATH = 'scene_cap_mask.png'
SCENE_ANNOTATIONS_PATH = 'scene_annotations.json'

# COCO annotation parameters
CATEGORY_ID = 1
CATEGORY_NAME = 'mushroom_cap'
SUPERCATEGORY = 'object'

# Seed for reproducibility
RANDOM_SEED = 42
