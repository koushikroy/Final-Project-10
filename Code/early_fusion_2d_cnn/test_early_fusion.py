#!/usr/bin/env python3
import os
import argparse
import logging
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score

# ─── COPY MODEL & DATASET CLASSES ─────────────────────────────────────────────
class EarlyFusionCNN(nn.Module):
    def __init__(self, num_frames, in_channels=3, height=224, width=224, num_classes=8):
        super().__init__()
        fused_ch = num_frames * in_channels
        self.conv1 = nn.Conv2d(fused_ch,  64, 3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,        128,3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,       256,3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,       512,3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(512)

        h4, w4 = height//4, width//4
        flat_size = 512 * h4 * w4
        self.fc1 = nn.Linear(flat_size, 512)
        self.drop= nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,       1024)
        self.fc3 = nn.Linear(1024,      num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        for conv, bn in [(self.conv1,self.bn1),
                         (self.conv2,self.bn2),
                         (self.conv3,self.bn3),
                         (self.conv4,self.bn4)]:
            x = nn.functional.relu(bn(conv(x)))
        x = x.flatten(1)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

class FrameDataset(Dataset):
    def __init__(self, root_dir, split, num_frames, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.num_frames = num_frames
        classes = sorted(os.listdir(os.path.join(root_dir, split)))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            cdir = os.path.join(root_dir, split, c)
            for sid in os.listdir(cdir):
                samp = os.path.join(cdir, sid)
                if os.path.isdir(samp):
                    self.samples.append(samp)
                    self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samp_dir = self.samples[idx]
        label = self.labels[idx]
        frames = sorted([os.path.join(samp_dir, f)
                         for f in os.listdir(samp_dir) if f.endswith('.jpg')])
        idxs = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
        imgs = []
        for i in idxs:
            img = Image.open(frames[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        clip = torch.stack(imgs, dim=0)
        return clip, label, samp_dir

# ─── MAIN TEST FUNCTION ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to best_model.pth from training")
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='output_test')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'log_test.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])
    logging.info("TESTING EARLY FUSION")
    logging.info(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # data & loader
    test_ds = FrameDataset(args.data_root, 'test', args.num_frames, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # load model
    model = EarlyFusionCNN(args.num_frames).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels= []
    misclassified = []  # (sample_dir, true, pred)

    with torch.no_grad():
        for clips, labels, samp_dirs in tqdm(test_loader, desc="Testing"):
            clips = clips.to(device)
            outputs = model(clips)
            preds = outputs.argmax(1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            # record misclassified
            for d, t, p in zip(samp_dirs, labels, preds):
                if t != p and len(misclassified) < 25:  # up to 25 samples
                    misclassified.append((d, t, p))

    # overall metrics
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

    # ─── save confusion matrix plot ─────────────────────────
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'Confusion Matrix (acc={acc:.4f})')
    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))

    # ─── class-wise accuracy bar plot ───────────────────────
    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(8,4))
    plt.bar(range(len(class_acc)), class_acc)
    plt.xticks(range(len(class_acc)), [str(i) for i in range(len(class_acc))])
    plt.ylabel('Accuracy')
    plt.xlabel('Class index')
    plt.title('Per-class Accuracy')
    plt.savefig(os.path.join(args.out_dir, 'class_accuracy.png'))

    # ─── misclassified samples grid ─────────────────────────
    n_show = min(len(misclassified), 5)
    fig, axes = plt.subplots(n_show, 5, figsize=(15, 3*n_show))
    for row, (d, t, p) in enumerate(misclassified[:n_show]):
        frames = sorted([os.path.join(d,f) for f in os.listdir(d) if f.endswith('.jpg')])
        idxs = np.linspace(0, len(frames)-1, 5, dtype=int)
        for col, i in enumerate(idxs):
            img = Image.open(frames[i]).convert('RGB')
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        axes[row,0].set_title(f"True:{t} Pred:{p}", loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'misclassified.png'))

    logging.info("Testing complete.")

if __name__ == '__main__':
    from tqdm import tqdm
    main()
