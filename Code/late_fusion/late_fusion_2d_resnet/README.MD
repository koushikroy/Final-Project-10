# Late Fusion 2D ResNet for Video Classification

---

## Description
This project implements a Late Fusion 2D ResNet model for video classification. The model processes video frames as input, extracts spatial features using a ResNet backbone, and performs late fusion by aggregating features across frames. The dataset consists of extracted frames from videos, organized into `train`, `val`, and `test` splits.

---

## Overview
The project includes the following functionalities:
1. **Training**: Train the Late Fusion 2D ResNet model on the extracted frames dataset.
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
python train.py --data_root "dataset" --num_frames 10 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "Code/late_fusion/late_fusion_2d_resnet/results/frames_10/augment" --augment
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
python test.py --data_root "dataset" --model_path "Code/late_fusion/late_fusion_2d_resnet/results/frames_10/augment/best_model.pth" --num_frames 10 --batch_size 16 --out_dir "Code/late_fusion/late_fusion_2d_resnet/results/frames_10/augment"
```
- `--data_root`: Path to the dataset directory.
- `--model_path`: Path to the saved model file from training.
- `--num_frames`: Number of frames to use per video.
- `--batch_size`: Batch size for testing.
- `--out_dir`: Directory to save the results.

---

## Output
- **Training**:
  - Best model: `best_model.pth`
  - Training logs: `log_train.txt`
  - Accuracy and loss curves: `acc_curve.png`, `loss_curve.png`
- **Testing**:
  - Test logs: `log_test.txt`
  - Confusion matrix: `confusion_matrix.png`
  - Class-wise accuracy: `class_accuracy.png`

---

## Notes
- Ensure all dependencies are installed before running the scripts. Use the following command to install required Python packages:
  ```bash
  pip install -r requirements.txt
  ```
- Modify paths in the scripts if your dataset is located in a different directory.
