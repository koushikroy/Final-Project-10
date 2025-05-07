# Final-Project-10
Final project for Deep Learning course (OSU)

This project explores visual-only emotion recognition in videos using deep neural network architectures. Leveraging the RAVDESS dataset, it implements several models, including early fusion, late fusion, 3D convolutional networks, and CNN–RNN hybrids, to compare their performance and analyze the impact of temporal modeling, data augmentation, and hyperparameter choices. The dataset has been preprocessed, with 16 frames per clip saved as zipped files. Users are required to unzip the dataset and then run the provided training and testing scripts to evaluate the models.

---

## Dataset Preparation
The extracted frames dataset is provided as zipped files. To use the dataset:
1. Unzip the dataset into the desired directory. Run the following codes (from the project root directory).
   ```bash
   python Code\utils\unzip.py dataset_zipped\train -o dataset\train 
   python Code\utils\unzip.py dataset_zipped\test -o dataset\test 
   python Code\utils\unzip.py dataset_zipped\val -o dataset\val 
   ```
2. Ensure the dataset directory is organized as follows:
   ```
   dataset/
   ├── train/
   ├── val/
   └── test/
   ```

---

## Training and Testing Instructions
The project includes four different model architectures:
1. **Early Fusion** (`Code/early_fusion`)
2. **Late Fusion** (`Code/late_fusion`)
3. **3D CNN** (`Code/3d_cnn`)
4. **CNN-RNN** (`Code/cnn_rnn`)

Each folder contains its own training and testing scripts. The instructions for running the scripts are the same across all folders.

### Steps to Train and Test
1. **Train the Model**:
   Run the training script with the following command (from the project root dir):
   ```bash
   python Code\3d_cnn\3d_cnn_resnet\train.py --data_root "dataset" --num_frames 10 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "Code\3d_cnn\3d_cnn_resnet\results\frames_10\augment" --augment
   ```
   - `--data_root`: Path to the dataset directory.
   - `--augment`: Optional flag to enable data augmentation during training.

2. **Test the Model**:
   After training, run the testing script with the following command (from the project root dir):
   ```bash
   python Code\3d_cnn\3d_cnn_resnet\test.py  --data_root "dataset" --model_path "Code\3d_cnn\3d_cnn_resnet\results\frames_10\augment\best_model.pth" --num_frames 10 --batch_size 16 --out_dir "Code\3d_cnn\3d_cnn_resnet\results\frames_10\augment"
   ```
   - `--data_root`: Path to the dataset directory.
   - `--model_path`: Path to the saved model file from training.

---

## Notes
- Ensure all dependencies are installed before running the scripts. Use the following command to install required Python packages:
  ```bash
  pip install -r requirements.txt
  ```
- Modify paths in the scripts if your dataset is located in a different directory.
- The training and testing scripts in all four folders (`early_fusion`, `late_fusion`, `3d_cnn`, `cnn_rnn`) follow the same structure and commands.
- Run the python codes directly from root dir. If you run by moving to the folder that the code is placed, you will run into issues with the other relative directories. In that case you will be better off with using the absolute paths in the code arguments while running.
---

## Output
- **Training**:
  - Best model: `output/best_model.pth`
  - Training logs: `output/log_train.txt`
  - Accuracy and loss curves: `output/acc_curve.png`, `output/loss_curve.png`
- **Testing**:
  - Test logs: `output/log_test.txt`
  - Confusion matrix: `output/confusion_matrix.png`