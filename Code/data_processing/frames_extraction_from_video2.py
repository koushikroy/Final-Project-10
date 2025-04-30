# Intelligent resizing (224 by 224) and extracting 16 frames from each video in the dataset
# Resize shorter size to (256 by 256) + CenterCrop(224)

import os
import zipfile
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# === User‐configurable paths ===
raw_dataset_path    = r'1188976'   # folder with Video_Speech_Actor_XX.zip
output_dataset_path = r'dataset'   # root for train/val/test folders

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

# === Helpers for Resize(256) + CenterCrop(224) ===
def resize_shorter_side(img, target_size=256):
    h, w = img.shape[:2]
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def center_crop(img, crop_size=224):
    h, w = img.shape[:2]
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

# === 1) Gather all videos ===
all_videos = []  # (zip_path, file_within_zip, emotion, actor_id)
for zip_name in os.listdir(raw_dataset_path):
    if not zip_name.startswith('Video_Speech_Actor_') or not zip_name.endswith('.zip'):
        continue
    zip_path = os.path.join(raw_dataset_path, zip_name)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if not member.lower().endswith('.mp4'):
                continue
            video_name = os.path.basename(member)
            parts = video_name[:-4].split('-')
            emo_code = parts[2]
            emo_name = emotion_map[emo_code]
            actor_id = int(parts[6])
            all_videos.append((zip_path, member, emo_name, actor_id))

# === 2) Actor → split assignment ===
def actor_to_split(actor_id):
    if 1 <= actor_id <= 20:
        return 'train'
    elif 21 <= actor_id <= 22:
        return 'val'
    else:
        return 'test'

video_list = [
    (zip_path, member, emo, actor_to_split(actor_id))
    for zip_path, member, emo, actor_id in all_videos
]

# === 3) Group by zip to minimize repeated extraction ===
videos_by_zip = defaultdict(list)
for zip_path, member, emo, split in video_list:
    videos_by_zip[zip_path].append((member, emo, split))

# === 4) Create output folders ===
for split in ('train', 'val', 'test'):
    for emo in emotion_map.values():
        os.makedirs(os.path.join(output_dataset_path, split, emo), exist_ok=True)

# Temporary extraction dir
tmp_dir = os.path.join(raw_dataset_path, 'tmp_extracted')

# === 5) Extract frames per video ===
for zip_path, entries in tqdm(videos_by_zip.items(), desc="Processing zips"):
    # clear tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # unzip all videos in this archive
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

    # process each video
    for member, emo, split in entries:
        video_file = os.path.join(tmp_dir, member)
        sample_id  = os.path.basename(member)[:-4]
        dest_dir   = os.path.join(output_dataset_path, split, emo, sample_id)
        os.makedirs(dest_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # sample 16 evenly spaced frames
        indices = np.linspace(0, total_frames - 1, 16, dtype=int)

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply Resize(256) + CenterCrop(224)
            img_resized = resize_shorter_side(frame, target_size=256)
            img_cropped = center_crop(img_resized, crop_size=224)

            out_path = os.path.join(dest_dir, f"frame_{i+1:04d}.jpg")
            cv2.imwrite(out_path, img_cropped)

        cap.release()

    # clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

print("✅ Done! 16 frames per video extracted with Resize(256) + CenterCrop(224), organized by split.")  
