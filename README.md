# SMS3D: 3D Synthetic Mushroom Scenes Dataset for Object Detection and Pose Estimation

## Introduction
SMS3D is a synthetic dataset designed for training and evaluating computer vision models in mushroom detection, instance segmentation, and pose estimation. It includes 40,000 unique scenes featuring white Agaricus bisporus and brown baby bella mushrooms in varied positions, orientations, and growth stages. The dataset supports applications such as automated harvesting, growth monitoring, and quality assessment.

This repository contains the data generation pipeline and pose estimation code. The synthetic dataset is highly customizable, allowing researchers to extend it based on their requirements.

**Key Features:**
- 40,000 annotated RGB-D scenes.
- Pose estimation using 6D rotation representation with geodesic loss.
- Benchmark for mushroom detection, segmentation, and pose estimation.

## Instructions to Run the Code

### Step 1: Create a Conda Environment
First, create a new Conda environment using the provided `env.yml` file:
```bash
conda env create -f env.yml
```

### Step 2: Activate the Environment
Activate the newly created environment:
```bash
conda activate sms3d-env
```

### Step 3: Run the Code
To generate synthetic scenes or train the pose estimation model, run `main.py` with the desired arguments. Below is an example:
```bash
python main.py --start_scene 1 --num_scenes 23000 --num_gpus 8
```

### Command-Line Arguments
The script accepts the following arguments:
```python
parser = argparse.ArgumentParser(description='Generate synthetic scenes.')
parser.add_argument('--start_scene', type=int, default=1, help='Starting scene number.')
parser.add_argument('--num_scenes', type=int, default=1, help='Number of scenes to generate.')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
```
- `--start_scene`: Specify the starting scene number.
- `--num_scenes`: Define how many scenes to generate.
- `--num_gpus`: Set the number of GPUs for processing.

## Example Config File
The `config.py` file allows customization of the scene generation and model parameters. Below is a sample configuration:
```python
import os

# Directories
PROJECT_ROOT = '.'  # Base directory
SOIL_DIR = os.path.join(PROJECT_ROOT, 'soil')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'white')

# Noise Parameters
NOISE_SCALE = 1
PERLIN_SCALE = 0.5
OCTAVES = 6
PERSISTENCE = 0.5
LACUNARITY = 2.0

# Ground Plane Parameters
GROUND_I_SIZE = 50
GROUND_J_SIZE = 50
GROUND_I_RESOLUTION = 200
GROUND_J_RESOLUTION = 200

# Mushroom Parameters
TOTAL_MUSHROOMS = 75
ANGLE_RANGE = (-90, 90)
SCALE_RANGE = (0.8, 2)
STRETCH_RANGE = (0.8, 1.2)

# Rendering Parameters
WINDOW_SIZE = [1024, 1024]
BACKGROUND_COLOR = 'white'
```

To customize the dataset:
1. Modify `TOTAL_MUSHROOMS` for the number of mushrooms per scene.
2. Adjust `NOISE_SCALE` and `PERLIN_SCALE` for terrain variability.
3. Update `MODELS_DIR` to use different 3D mushroom models.

## Dataset and Models
- The 3D mushroom models and high-resolution soil images are not included in this repository due to copyright restrictions. These resources will be provided to researchers upon request. Please contact Abdollah Zakeri at [azakeri@uh.edu](mailto:azakeri@uh.edu).
- Links to the generated scenes and pre-trained pose estimation models will be posted here shortly.

## Citation
If you use this dataset or code in your research, please cite:
```text
Zakeri, A.; Koirala, B.; Kang, J.; Balan, V.; Zhu, W.; Benhaddou, D.; Merchant, F.A. SMS3D: 3D Synthetic Mushroom Scenes Dataset for Object Detection and Pose Estimation. Computers, 2024.
```

---
We hope SMS3D accelerates advancements in computer vision for agricultural automation. Feel free to open issues or pull requests for any contributions or questions!
