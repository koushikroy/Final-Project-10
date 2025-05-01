#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# --- Dataset ---
class LateFusionDataset(Dataset):
    def __init__(self, root_dir, split, num_frames, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.num_frames = num_frames
        classes = sorted(os.listdir(os.path.join(root_dir, split)))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            for sid in os.listdir(os.path.join(root_dir, split, cls)):
                samp_dir = os.path.join(root_dir, split, cls, sid)
                if os.path.isdir(samp_dir):
                    self.samples.append(samp_dir)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samp_dir = self.samples[idx]
        label = self.labels[idx]
        frames = sorted([os.path.join(samp_dir, f) for f in os.listdir(samp_dir) if f.endswith('.jpg')])
        idxs = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        imgs = []
        for i in idxs:
            img = Image.open(frames[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        clip = torch.stack(imgs, dim=0)  # (T, C, H, W)
        return clip, label, samp_dir

# --- Model ---
class LateFusionResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, -1)
        agg = feats.mean(dim=1)
        return self.fc(agg)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--out_dir', default='output_test')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(args.out_dir, 'log_test.txt')),
                            logging.StreamHandler()
                        ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        normalize
    ])

    test_ds = LateFusionDataset(args.data_root, 'test', args.num_frames, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LateFusionResNet(len(test_ds.class_to_idx)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels, _ in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            preds = model(clips).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

    class_names = list(sorted(test_ds.class_to_idx, key=lambda k: test_ds.class_to_idx[k]))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2. else 'black')
    plt.title(f"Confusion Matrix (acc={acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))

    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure()
    plt.bar(range(len(class_names)), class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Per-Class Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'class_accuracy.png'))

if __name__ == '__main__':
    main()
