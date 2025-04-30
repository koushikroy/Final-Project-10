# Data Processing for Deep Learning Final Project

This directory contains scripts for processing and analyzing datasets for the Deep Learning Final Project. The scripts handle tasks such as extracting frames from videos, organizing datasets into train/validation/test splits, and generating statistics and visualizations for the dataset.

## Contents

### 1. `frames_extraction_from_video_1.py`
- **Purpose**: Extracts 16 evenly spaced frames from each video in the dataset and resizes them to `224x224` using naive resizing.
- **Key Features**:
  - Stratified splitting of videos into `train`, `val`, and `test` sets.
  - Organizes extracted frames into folders based on emotion and split.
  - Uses `scikit-learn` for stratified splitting.

### 2. `frames_extraction_from_video2.py`
- **Purpose**: Extracts 16 evenly spaced frames from each video in the dataset and applies intelligent resizing (`Resize(256)` + `CenterCrop(224)`).
- **Key Features**:
  - Assigns videos to splits (`train`, `val`, `test`) based on actor IDs.
  - Organizes extracted frames into folders based on emotion and split.
  - Applies advanced resizing techniques for better frame quality.

### 3. `print_train_test_val_stats.py`
- **Purpose**: Analyzes the dataset and generates statistics and visualizations for the `train`, `val`, and `test` splits.
- **Key Features**:
  - Counts the number of samples per emotion and actor for each split.
  - Generates bar plots for the distribution of samples per emotion and actor.
  - Saves plots as PNG files in the current directory.

## Usage

### Frame Extraction
1. Place the raw dataset (e.g., `Video_Speech_Actor_XX.zip` files) in the directory specified by `raw_dataset_path`.
2. Configure the `raw_dataset_path` and `output_dataset_path` variables in the script.
3. Run either `frames_extraction_from_video_1.py` or `frames_extraction_from_video2.py` depending on the resizing method you prefer.
4. Extracted frames will be saved in the `output_dataset_path` directory, organized by split and emotion.

### Dataset Statistics
1. Ensure the dataset is processed and organized into `train`, `val`, and `test` splits.
2. Configure the `output_dataset_path` variable in `print_train_test_val_stats.py`.
3. Run the script to generate statistics and visualizations.
4. Check the console output for statistics and the current directory for saved plots.
5. Run `cd Code\data_processing` then `python print_train_test_val_stats.py`

## Requirements
- Python 3.7+
- Required Python packages:
  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `tqdm`
  - `scikit-learn` (for `frames_extraction_from_video_1.py`)

## Output
- **Frame Extraction**:
  - Extracted frames organized in the following structure:
    ```
    dataset/
    ├── train/
    │   ├── emotion_1/
    │   │   ├── sample_1/
    │   │   │   ├── frame_0001.jpg
    │   │   │   ├── ...
    │   ├── ...
    ├── val/
    ├── test/
    ```
- **Statistics and Visualizations**:
  - Console output with sample counts per emotion and actor.
  - Bar plots saved as:
    - `emotion_distribution_[split].png`
    - `actor_distribution_[split].png`

## Notes
- Ensure the raw dataset is properly formatted and matches the expected structure.
- Modify paths and parameters in the scripts as needed for your specific setup.

## License
This project is for educational purposes and follows the applicable licensing terms of the dataset used.
