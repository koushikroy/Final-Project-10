#!/usr/bin/env python3
import os
import time
import argparse
import logging
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# ─── 1) MODEL DEFINITION ──────────────────────────────────────────────────────
class EarlyFusionCNN(nn.Module):
    def __init__(self, num_frames, in_channels=3, height=224, width=224, num_classes=8):
        super().__init__()
        fused_ch = num_frames * in_channels

        # four conv blocks
        self.conv1 = nn.Conv2d(fused_ch,  64,  kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,        128, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,       256, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,       512, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(512)

        # compute flattened feature size
        h4, w4 = height // 4, width // 4
        flat_size = 512 * h4 * w4

        # classifier head
        self.fc1 = nn.Linear(flat_size, 512)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512,       1024)
        self.fc3 = nn.Linear(1024,      num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)   # early fusion

        # four conv blocks
        for conv, bn in [(self.conv1, self.bn1),
                         (self.conv2, self.bn2),
                         (self.conv3, self.bn3),
                         (self.conv4, self.bn4)]:
            x = nn.functional.relu(bn(conv(x)))

        x = x.flatten(1)                   # (B, flat_size)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)                    # (B, num_classes)
        return x

# ─── 2) CUSTOM DATASET ────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, root_dir, split, num_frames, transform=None):
        """
        root_dir/
          train/val/test/
            happy/neutral/.../
               01-01-02-.../   <- sample_id folders
                  frame_0001.jpg ...
        """
        self.samples = []
        self.labels  = []
        self.transform = transform
        self.num_frames = num_frames

        # build class→index map
        classes = sorted(os.listdir(os.path.join(root_dir, split)))
        self.class_to_idx = {cls:i for i,cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, split, cls)
            for sid in os.listdir(cls_dir):
                samp_dir = os.path.join(cls_dir, sid)
                if os.path.isdir(samp_dir):
                    self.samples.append(samp_dir)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samp_dir = self.samples[idx]
        label = self.labels[idx]

        # list all frames
        frames = sorted([
            os.path.join(samp_dir, f)
            for f in os.listdir(samp_dir)
            if f.endswith('.jpg')
        ])
        # pick equidistant indices
        total = len(frames)
        idxs = np.linspace(0, total-1, self.num_frames, dtype=int)

        imgs = []
        for i in idxs:
            img = Image.open(frames[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # stack to (T, C, H, W)
        clip = torch.stack(imgs, dim=0)
        return clip, label

# ─── 3) TRAIN / VAL LOOP ─────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for clips, labels in tqdm(loader, desc="  Train batches", leave=False):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="  Val batches", leave=False):
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# ─── 4) MAIN ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help="Path to dataset root containing train/val/test")
    parser.add_argument('--num_frames', type=int, default=16,
                        help="How many frames to sample per clip (<=16)")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=5,
                        help="Early stopping patience on val-acc")
    parser.add_argument('--out_dir', type=str, default='output',
                        help="Where to save logs, models, plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # setup logging to both console & file
    log_path = os.path.join(args.out_dir, 'log_train.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])
    logging.info("TRAINING EARLY FUSION")
    logging.info(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # datasets & loaders
    train_ds = FrameDataset(args.data_root, 'train', args.num_frames, transform)
    val_ds   = FrameDataset(args.data_root,   'val', args.num_frames, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # model, loss, optimizer, scheduler
    model = EarlyFusionCNN(args.num_frames).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=0.5, patience=2,
                                  verbose=True)

    # trackers
    history = defaultdict(list)
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader,
                                            criterion, optimizer, device)
        val_loss,   val_acc   = validate_epoch(model, val_loader,
                                               criterion, device)
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        logging.info(f"Epoch {epoch:02d} | "
                     f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                     f"Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f} | "
                     f"Time: {elapsed:.1f}s")

        # early stopping & checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  → New best model saved at epoch {epoch}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info("Early stopping triggered.")
                break

    # final summary
    logging.info("Training complete")
    logging.info(f"Best Val Acc: {best_val_acc:.4f}")
    logging.info(f"Model saved to: {ckpt_path}")

    # save plots
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(); plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'],   label='val_loss'); plt.legend()
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.savefig(os.path.join(args.out_dir, 'loss_curve.png'))

    plt.figure(); plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'],   label='val_acc'); plt.legend()
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.savefig(os.path.join(args.out_dir, 'acc_curve.png'))

if __name__ == '__main__':
    main()
