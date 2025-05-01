# test.py
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
from torchvision.models.video import r3d_18
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Custom Dataset ---
class FrameDataset(Dataset):
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
        idxs = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
        imgs = []
        for i in idxs:
            img = Image.open(frames[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        clip = torch.stack(imgs, dim=1)
        return clip, label, samp_dir

# --- Model Definition ---
class VideoResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = r3d_18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# --- Main Testing Function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--out_dir', default='output_test')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'log_test.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info("====== TESTING MODEL ======")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
    transform   = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        normalize,
    ])

    test_ds = FrameDataset(args.data_root, 'test', args.num_frames, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = VideoResNet(num_classes=len(test_ds.class_to_idx), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds, all_labels, misclassified = [], [], []
    with torch.no_grad():
        for clips, labels, paths in test_loader:
            clips = clips.to(device)
            outputs = model(clips)
            preds = outputs.argmax(1).cpu().numpy()
            labels_np = labels.numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())
            for p, t, d in zip(preds, labels_np, paths):
                if p != t and len(misclassified) < 5:
                    misclassified.append((d, t, p))

    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    class_names = list(sorted(test_ds.class_to_idx, key=lambda k: test_ds.class_to_idx[k]))

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix (acc={acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))

    # Plot per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure()
    plt.bar(range(len(class_acc)), class_acc)
    plt.xticks(range(len(class_acc)), class_names, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Per-Class Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'class_accuracy.png'))

    logging.info("Testing complete.")

if __name__ == '__main__':
    main()
