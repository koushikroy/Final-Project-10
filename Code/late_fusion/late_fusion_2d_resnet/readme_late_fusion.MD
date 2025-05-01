# Late Fusion Video Classification

This repository implements a **late fusion** architecture for video classification using a pretrained `r3d_18` backbone. In late fusion, the input clip is split into temporal segments, each processed independently, and their predictions are combined at the end.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Pipeline](#data-pipeline)
- [Model Components](#model-components)
  - [Backbone: R3D-18](#backbone-r3d-18)
  - [Late Fusion Module](#late-fusion-module)
- [Training](#training)
- [Testing](#testing)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

---

## Architecture Overview

1. **Input Clip**: A sequence of `T` frames of shape `(C, T, H, W)`.
2. **Segmentation**: The clip is divided into `S` equal temporal segments, each with `T/S` frames.
3. **Frame-Level Processing**: Each segment is fed independently into the same pretrained 3D ResNet-18 (`r3d_18`) backbone.
4. **Per-Segment Prediction**: The backbone outputs logits of shape `(B, num_classes)` for each segment.
5. **Fusion**: Logits from all `S` segments are averaged to produce the final prediction per clip.

This strategy preserves temporal locality while still leveraging global context at decision time.

---

## Data Pipeline

- **FrameDataset**: Custom `torch.utils.data.Dataset` that:
  - Reads frames from a directory structure: `root/{split}/{class}/{sample_id}/*.jpg`
  - Samples `T` frames uniformly from each clip
  - Applies optional augmentation during training (resize, crop, flip)
  - Stacks frames into a tensor of shape `(C, T, H, W)`

- **Transforms**:
  - **Training**:
    - Resize to 128×128
    - Random crop 112×112
    - Random horizontal flip (if `--augment`)
    - Normalize with ImageNet video statistics
  - **Validation/Test**:
    - Resize to 128×128
    - Center crop 112×112
    - Normalize

---

## Model Components

### Backbone: R3D-18

- Pretrained 3D ResNet-18 for video
- Expects input shape `(B, C, T_seg, H, W)`.
- Final fully-connected layer replaced to match `num_classes`.

### Late Fusion Module

```python
class LateFusionVideoResNet(nn.Module):
    def __init__(self, num_classes, num_segments=4, pretrained=True):
        # Load pretrained 3D ResNet-18
        # Replace final FC
        # Store number of segments

    def forward(self, x):
        B, C, T, H, W = x.shape
        seg_size = T // self.num_segments
        logits = []
        for i in range(self.num_segments):
            seg = x[:, :, i*seg_size:(i+1)*seg_size, :, :]
            out = self.backbone(seg)
            logits.append(out)
        logits = torch.stack(logits, dim=1)  # (B, S, num_classes)
        return logits.mean(dim=1)           # (B, num_classes)
```

- **Pros**:
  - Maintains segment-level specialization
  - Simple aggregation via average
- **Cons**:
  - Ignores interactions across segments
  - Assumes uniform importance of each segment

---

## Training

Run:
```bash
python train.py \
  --data_root /path/to/data \
  --num_frames 16 \
  --segments 4 \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --wd 1e-4 \
  --patience 5 \
  --num_workers 4 \
  --out_dir output \
  [--augment]
```

- Model checkpoints and logs saved to `output/`.
- Early stopping after `patience` epochs without val improvement.

---

## Testing

Run:
```bash
python test.py \
  --data_root /path/to/data \
  --model_path output/best_model.pth \
  --num_frames 16 \
  --segments 4 \
  --batch_size 8 \
  --out_dir output_test
```

- Outputs test accuracy, confusion matrix, per-class accuracy plots.

---

## Usage

1. Prepare data directory:
   ```
   root/
     train/
       class0/
         sample0/*.jpg
         sample1/*.jpg
       class1/...
     val/
     test/
   ```
2. Install requirements.
3. Train and evaluate as above.

---

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- TorchVision >= 0.11
- NumPy, Pillow, Scikit-learn, Matplotlib

Install with:
```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib
```

---

## License

MIT License
