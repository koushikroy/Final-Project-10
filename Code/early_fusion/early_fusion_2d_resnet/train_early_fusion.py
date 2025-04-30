#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # <-- ADD THIS FIRST
import time
import argparse
import logging
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # <-- ADD THIS LINE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# --- Model Definition ---
class EarlyFusionCNN(nn.Module):
    def __init__(self, num_frames, in_channels=3, num_classes=8):
        super().__init__()
        fused_ch = num_frames * in_channels

        self.conv1 = nn.Conv2d(fused_ch, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        for conv, bn in [(self.conv1, self.bn1),
                         (self.conv2, self.bn2),
                         (self.conv3, self.bn3),
                         (self.conv4, self.bn4)]:
            x = nn.functional.relu(bn(conv(x)))

        x = self.global_pool(x)
        x = x.view(B, -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

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

        frames = sorted(f for f in os.listdir(samp_dir) if f.endswith('.jpg'))
        total = len(frames)
        idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)

        imgs = []
        for i in idxs:
            img = Image.open(os.path.join(samp_dir, frames[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        clip = torch.stack(imgs, dim=0)
        return clip, label

# --- Training and Validation Functions ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = total = 0
    for clips, labels in loader:
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
    correct = total = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_dir', default='output')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    log_file = os.path.join(args.out_dir, 'log_train.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("========== TRAINING STARTED ==========")
    logging.info(f"Parameters: {vars(args)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FrameDataset(args.data_root, 'train', args.num_frames, train_transform)
    val_ds = FrameDataset(args.data_root, 'val', args.num_frames, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EarlyFusionCNN(args.num_frames).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    epochs_no_improve = 0
    runtimes = []
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        logging.info(f"---------- Epoch {epoch}/{args.epochs} ----------")
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        epoch_time = time.time() - start
        runtimes.append(epoch_time)
        avg_time = sum(runtimes) / len(runtimes)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logging.info(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        logging.info(f"Epoch Time: {epoch_time:.1f}s | Average Time: {avg_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"-> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("========== TRAINING COMPLETE ==========")
    logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logging.info(f"Model checkpoint at: {ckpt_path}")

    # Save plots
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.out_dir, 'loss_curve.png'))
    plt.close()  # <-- VERY IMPORTANT

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(args.out_dir, 'acc_curve.png'))
    plt.close()  # <-- VERY IMPORTANT

if __name__ == '__main__':
    main()
