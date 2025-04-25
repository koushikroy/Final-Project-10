import os
import zipfile
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

# === User-configurable paths ===
raw_dataset_path   = r'C:\Users\rkous\Downloads\Compressed\1188976'         # folder containing Video_Speech_Actor_XX.zip
output_dataset_path = r'C:\Users\rkous\Downloads\Compressed\dataset'  # root for train/val/test folders

# === Emotion code → name map ===
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprise',
}

# === 1) Collect all videos with their zip_path, file_path, emotion ===
all_videos = []
for zip_name in os.listdir(raw_dataset_path):
    if not zip_name.startswith('Video_Speech_Actor_') or not zip_name.endswith('.zip'):
        continue
    zip_path = os.path.join(raw_dataset_path, zip_name)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file in zf.namelist():
            if file.endswith('.mp4'):
                video_name = os.path.basename(file)
                parts = video_name[:-4].split('-')
                emo_code = parts[2]
                emo_name = emotion_map[emo_code]
                all_videos.append((zip_path, file, emo_name))

# === 2) Split into train/val/test stratified by emotion ===
X = [(zip_path, file_path) for zip_path, file_path, _ in all_videos]
y = [emo_name for _, _, emo_name in all_videos]

# First split: 70% train, 30% temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
# Second split: temp into 10% val (1/3 of 30%) and 20% test (2/3 of 30%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=42)

# Combine into a single list with split information
video_list = []
for X, y, split in [(X_train, y_train, 'train'), (X_val, y_val, 'val'), (X_test, y_test, 'test')]:
    for (zip_path, file_path), emo in zip(X, y):
        video_list.append((zip_path, file_path, emo, split))

# === 3) Group videos by zip_path ===
videos_by_zip = defaultdict(list)
for zip_path, file_path, emo, split in video_list:
    videos_by_zip[zip_path].append((file_path, emo, split))

# === 4) Create folder hierarchy ===
for split in ('train', 'val', 'test'):
    for emo in emotion_map.values():
        os.makedirs(os.path.join(output_dataset_path, split, emo), exist_ok=True)

# Temporary folder for unzipping
tmp_dir = os.path.join(raw_dataset_path, 'tmp_extracted')

# === 5) Process each zip file ===
for zip_path in tqdm(videos_by_zip.keys(), desc="Processing zips"):
    # Clear and recreate tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Unzip to tmp_dir
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)
    
    # Process each video in this zip
    for file_path, emo, split in videos_by_zip[zip_path]:
        video_file = os.path.join(tmp_dir, file_path)
        sample_id = os.path.basename(file_path)[:-4]
        dest_dir = os.path.join(output_dataset_path, split, emo, sample_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Open video and sample 16 frames evenly
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 16, dtype=int)
        
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Resize to 224x224
            frame_resized = cv2.resize(frame, (224, 224))
            out_path = os.path.join(dest_dir, f"frame_{i+1:04d}.jpg")
            cv2.imwrite(out_path, frame_resized)
        
        cap.release()
    
    # Cleanup tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

print("✅ Done! 16 frames per video have been extracted, resized, and organized.")