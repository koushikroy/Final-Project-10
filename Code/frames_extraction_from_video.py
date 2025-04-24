import os
import zipfile
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm

# === User-configurable paths ===
raw_dataset_path   = r'C:\Users\rkous\Downloads\Compressed\1188976'         # folder containing Video_Speech_Actor_XX.zip
output_dataset_path = r'C:\Users\rkous\OneDrive - Oklahoma A and M System\courses\Spring 25\Deep Learning\Projects\final_project\dataset'  # root for train/val/test folders

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

# === 1) Gather & split actor IDs randomly into 70/10/20 ===
all_actors = [f"{i:02d}" for i in range(1, 25)]  # '01' ... '24'
random.seed(42)
random.shuffle(all_actors)

n = len(all_actors)  # 24
train_cnt = round(0.7 * n)  # 17
val_cnt   = round(0.1 * n)  # 2
# remainder goes to test
train_ids = set(all_actors[:train_cnt])
val_ids   = set(all_actors[train_cnt:train_cnt + val_cnt])
test_ids  = set(all_actors[train_cnt + val_cnt:])

def get_split(actor_id):
    if   actor_id in train_ids: return 'train'
    elif actor_id in val_ids:   return 'val'
    else:                        return 'test'

# === 2) Create folder hierarchy ===
for split in ('train', 'val', 'test'):
    for emo in emotion_map.values():
        os.makedirs(os.path.join(output_dataset_path, split, emo), exist_ok=True)

# temporary folder for unzipping
tmp_dir = os.path.join(raw_dataset_path, 'tmp_extracted')
os.makedirs(tmp_dir, exist_ok=True)

# === 3) Loop through each speech zip, extract frames ===
for zip_name in tqdm(os.listdir(raw_dataset_path), desc="Zips"):
    if not (zip_name.startswith('Video_Speech_Actor_') and zip_name.endswith('.zip')):
        continue

    zip_path = os.path.join(raw_dataset_path, zip_name)

    # clear & recreate tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # unzip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

    # each zip contains one folder Actor_## 
    for actor_folder in os.listdir(tmp_dir):
        actor_path = os.path.join(tmp_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        # process all .mp4 files inside
        for vid_name in tqdm(os.listdir(actor_path),
                             desc=f" Actor {actor_folder}",
                             leave=False):
            if not vid_name.endswith('.mp4'):
                continue

            # parse filename: e.g. "02-01-06-01-02-01-12.mp4"
            parts = vid_name[:-4].split('-')
            modality, channel, emo_code, intensity, stmt, rep, actor_id = parts
            emo_name = emotion_map[emo_code]
            split    = get_split(actor_id)

            # create sample-specific output dir
            sample_id = vid_name[:-4]
            dest_dir = os.path.join(output_dataset_path, split, emo_name, sample_id)
            os.makedirs(dest_dir, exist_ok=True)

            # open video, sample 16 frames evenly
            video_file = os.path.join(actor_path, vid_name)
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total_frames - 1, 16, dtype=int)

            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                # resize to 224×224
                frame_resized = cv2.resize(frame, (224, 224))
                out_path = os.path.join(dest_dir, f"frame_{i+1:04d}.jpg")
                cv2.imwrite(out_path, frame_resized)

            cap.release()

# cleanup
shutil.rmtree(tmp_dir, ignore_errors=True)

print("✅ Done! 16 frames per video have been extracted, resized, and organized.")
