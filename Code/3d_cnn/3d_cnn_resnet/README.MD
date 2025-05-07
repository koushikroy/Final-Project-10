# 3D CNN ResNet for Video Classification

---

## Description
This project implements a 3D CNN ResNet model for video classification. The model processes video frames as input and predicts the corresponding class. The dataset consists of extracted frames from videos, organized into `train`, `val`, and `test` splits.

---

## Overview
The project includes the following functionalities:
1. **Training**: Train the 3D CNN ResNet model on the extracted frames dataset.
2. **Testing**: Evaluate the trained model on the test set and generate metrics such as accuracy and confusion matrix.
3. **Results**: Save the best model, training logs, and visualizations (e.g., accuracy/loss curves, confusion matrix).

---

## Dataset Structure
Ensure the dataset is organized as follows:
```
dataset/
├── train/
├── val/
└── test/
```

Each split contains subfolders for each class, and each class folder contains subfolders with extracted frames for each video.

---

## How to Run

### Train the Model
Run the training script with the following command:
```bash
python train.py --data_root "dataset" --num_frames 10 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "Code/3d_cnn/3d_cnn_resnet/results/frames_10/augment" --augment
```
- `--data_root`: Path to the dataset directory.
- `--num_frames`: Number of frames to use per video.
- `--epochs`: Maximum number of training epochs.
- `--patience`: Early stopping patience.
- `--batch_size`: Batch size for training.
- `--num_workers`: Number of workers for data loading.
- `--out_dir`: Directory to save the results.
- `--augment`: Enable data augmentation during training.

### Test the Model
After training, run the testing script with the following command:
```bash
python test.py --data_root "dataset" --model_path "Code/3d_cnn/3d_cnn_resnet/results/frames_10/augment/best_model.pth" --num_frames 10 --batch_size 16 --out_dir "Code/3d_cnn/3d_cnn_resnet/results/frames_10/augment"
```
- `--data_root`: Path to the dataset directory.
- `--model_path`: Path to the saved model

This project implements a 3D CNN ResNet model for video classification. The model processes video frames as input and predicts the corresponding class. The dataset consists of extracted frames from videos, organized into `train`, `val`, and `test` splits.

---

