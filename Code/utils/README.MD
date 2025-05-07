# Utils for Final-Project-10

This folder contains utility scripts for managing and processing the dataset used in the Final-Project-10. These scripts help with tasks such as zipping subfolders, unzipping archives, and comparing directories.

---

## Scripts Overview

### 1. `zip.py`
- **Description**: Zips each subfolder in a given directory into separate `.zip` archives.
- **Usage**:
  ```bash
  python zip.py <folder>
  ```
  - `<folder>`: Path to the directory containing subfolders to zip.
- **Example**:
  ```bash
  python zip.py dataset/train
  ```
  This will create `.zip` files for each subfolder in train.

---

### 2. unzip.py
- **Description**: Extracts all `.zip` files in a given directory into separate folders.
- **Usage**:
  ```bash
  python unzip.py <folder> -o <output_folder>
  ```
  - `<folder>`: Path to the directory containing `.zip` files.
  - `-o <output_folder>`: (Optional) Directory where extracted folders will be placed. Defaults to the input folder.
- **Example**:
  ```bash
  python unzip.py dataset_zipped/train -o dataset/train
  ```
  This will extract all `.zip` files in train into train.

---

### 3. checker.py
- **Description**: Compares two directories to check if they are identical (same files, subdirectories, and file contents).
- **Usage**:
  ```bash
  python checker.py <folder1> <folder2>
  ```
  - `<folder1>`: Path to the first directory.
  - `<folder2>`: Path to the second directory.
- **Example**:
  ```bash
  python checker.py dataset/train dataset_backup/train
  ```
  This will compare train and `dataset_backup/train` and report if they are identical.

---

## Notes
- Ensure all dependencies are installed before running the scripts. Use the following command to install required Python packages:
  ```bash
  pip install -r requirements.txt
  ```
- Modify paths in the scripts if your dataset is located in a different directory.

---

## Output
- **`zip.py`**: Creates `.zip` files for each subfolder in the specified directory.
- **`unzip.py`**: Extracts `.zip` files into corresponding folders.
- **`checker.py`**: Prints whether the two directories are identical or different.
