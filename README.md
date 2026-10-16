# Phase-Guided MoE Routing for Deepfake Detection

**2026 Graduation Project — Division of Artificial Intelligence Engineering, Sookmyung Women's University**  

**Authors:** Joo-Yeon Kim, Yeon-Seo Kim
**Advisor:** Prof. Byung-Gyu Kim

A ViT-B/16 based deepfake detection framework that mitigates performance degradation under lossy compression (JPEG / video encoding) by integrating Phase-Only Reconstruction into Mixture-of-Experts routing and combining it with JPEG Compression Augmentation.

---

## Table of Contents

- [Motivation](#motivation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference and Evaluation](#inference-and-evaluation)
- [File Structure](#file-structure)
- [Module Descriptions](#module-descriptions)
- [References](#references)

---

## Motivation

Existing MoE-FFD determines expert routing solely from spatial token representations. When JPEG compression corrupts those spatial features, routing decisions become unreliable and detection performance drops sharply (c23 AUC 98.93% → c40 AUC 72.93%).

This work addresses the problem with two complementary approaches:

1. **Phase-Guided Adapter-MoE Routing** — Phase features extracted via 2D FFT are concatenated with spatial tokens as the Adapter-MoE gating input, so that structural information preserved in the phase can assist routing even when spatial features are degraded by compression.

2. **JPEG Compression Augmentation** — During training, JPEG compression with quality factor q sampled uniformly from [30, 70] is applied with probability 0.5, encouraging the model to learn compression-robust representations.

---

## Model Architecture

<img width="6583" height="2419" alt="그림2" src="https://github.com/user-attachments/assets/2d3f05e5-d789-4a2f-9ac1-45c5de44cc8c" />

```
Input image (224x224)
|
+-- [Phase Stream]
|   2D FFT -> Phase-Only Reconstruction -> Lightweight CNN -> GAP -> Linear -> z_freq (768-dim)
|
+-- [ViT Stream]
    ViT-B/16 patch embedding
    |
    +-- Transformer Block x 12
        +-- LoRA-MoE   (Self-Attention)
        +-- Adapter-MoE (FFN)
            +-- Gate input: Concat(x_global, z_freq) [1536-dim]
                            -> Noisy Top-k Routing
```

- **Backbone**: ViT-B/16 pretrained on ImageNet-21k
- **LoRA-MoE**: Parameter-efficient update of Self-Attention weights
- **Adapter-MoE**: FFN-side experts with phase-guided routing
- **PhaseModule**: Phase-Only Reconstruction -> 128-dim -> 768-dim projection

---

## Results

### Cross-Compression Generalization (trained on c23, tested on c40)

| Method | c23 AUC | c40 AUC |
|--------|---------|---------|
| Baseline (MoE-FFD) | 98.93 | 72.93 |
| + JPEG Augmentation | 97.83 | 76.21 |
| + Phase + Aug (Ours) | 98.13 | **76.64** |

### Robustness by JPEG Quality Factor (FF++ c23 validation set)

| Method | Q=100 | Q=80 | Q=60 |
|--------|-------|------|------|
| Baseline | 98.9 | 96.4 | 92.5 |
| + JPEG Aug | 98.3 | 97.0 | 95.8 |
| + Phase + Aug (Ours) | **98.7** | **97.9** | **97.3** |
| Phase w/o Recon (Ablation) | 86.4 | 83.5 | 81.3 |

---

## Environment Setup

### Requirements

| Item | Version |
|------|---------|
| Python | 3.8 or higher |
| PyTorch | 1.13.0 or higher |
| CUDA | 11.7 or higher recommended |
| GPU | Tested on NVIDIA RTX 3090 (24 GB) |

### Installation

```bash
git clone https://github.com/lyeonsl/Phase-Guided-Moe-Routing.git
cd Phase-Guided-Moe-Routing
pip install -r requirements.txt
```

### Key Packages

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `timm` | ViT-B/16 pretrained model loading |
| `albumentations` | JPEG compression augmentation |
| `opencv-python` | Image processing |
| `scikit-learn`, `scipy` | AUC and EER computation |
| `tensorboard` | Training log visualization |

---

## Dataset Preparation

### FaceForensics++ Directory Structure

```
data/FF++/
+-- c23_crop/
|   +-- train/
|   |   +-- real/<video_id>/<frame>.png
|   |   +-- fake/<video_id>/<frame>.png
|   +-- val/
|       +-- real/<video_id>/<frame>.png
|       +-- fake/<video_id>/<frame>.png
+-- c40_crop/
    +-- real/<video_id>/<frame>.png
    +-- fake/<video_id>/<frame>.png
```

- **c23**: Light compression — used for training and validation
- **c40**: Heavy compression — used for cross-compression generalization evaluation
- Frames should be face-cropped PNG or JPG images (224x224 recommended)

---

## Training

### Baseline

```bash
python train.py \
    --train_path data/FF++/c23_crop/train \
    --valid_path data/FF++/c23_crop/val \
    --model_dir models/baseline/ \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --device 0
```

### With JPEG Augmentation

```bash
python train.py \
    --train_path data/FF++/c23_crop/train \
    --valid_path data/FF++/c23_crop/val \
    --model_dir models/jpeg_aug/ \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --device 0 \
    --jpeg_aug
```

### Full Proposed Method (Phase + Augmentation)

```bash
python train.py \
    --train_path data/FF++/c23_crop/train \
    --valid_path data/FF++/c23_crop/val \
    --model_dir models/phase_aug/ \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --device 0 \
    --jpeg_aug \
    --use_phase
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--train_path` | — | Path to training data |
| `--valid_path` | — | Path to validation data |
| `--model_dir` | — | Directory to save checkpoints |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 3e-5 | Adam learning rate |
| `--device` | 0 | GPU index |
| `--jpeg_aug` | False | Enable JPEG compression augmentation |
| `--use_phase` | False | Enable PhaseModule |
| `--uncertainty_routing` | False | Enable entropy-based adaptive Top-k routing |
| `--entropy_thresh` | 1.0 | Entropy threshold for uncertainty routing |

### Training Configuration Summary

| Item | Value |
|------|-------|
| Optimizer | Adam (lr=3e-5, weight_decay=1e-5) |
| Scheduler | StepLR (step_size=5, gamma=0.5) |
| Loss | CrossEntropy Loss + MoE Load-Balancing Loss |
| Batch size | 32 |
| Epochs | 20 |
| GPU | NVIDIA RTX 3090 |

---

## Inference and Evaluation

### Cross-Compression Evaluation (c40)

```bash
python inference.py \
    --data_root data/FF++/c40_crop \
    --ckpt_path models/phase_aug/best_model.pkl \
    --model moeffd_phase \
    --device cuda:0
```

### Evaluation at Specific JPEG Quality (e.g., Q=60)

```bash
python inference.py \
    --data_root data/FF++/c23_crop/val \
    --ckpt_path models/phase_aug/best_model.pkl \
    --model moeffd_phase \
    --device cuda:0 \
    --jpeg_quality 60
```

### Output

Results are saved to `results/results.json`:

```json
{
  "ACC": 0.6633,
  "AUC": 0.7664,
  "EER": 0.2981
}
```

An ROC curve image is also generated alongside the JSON file.

---

## File Structure

```
Phase-Guided-Moe-Routing/
+-- ViT_MoE_phase_aug.py   # Model definition
+-- train.py               # Training script
+-- dataset.py             # Dataset classes
+-- inference.py           # Inference and evaluation script
+-- utils.py               # Evaluation metrics (ACC, AUC, EER, APCER, BPCER, ACER)
+-- requirements.txt       # Package list
+-- README.md
```

---

## Module Descriptions

### `ViT_MoE_phase_aug.py`

#### PhaseModule

Extracts phase-based structural features from the input image using 2D FFT.

```
Input x (B, C, H, W)
-> 2D FFT: torch.fft.fft2(x, norm='ortho')
-> Phase extraction: torch.angle(fft)
-> Phase-Only Reconstruction: torch.exp(1j * phase) -> IFFT -> .real
-> Channel-wise normalization: (x - mean) / std
-> Lightweight CNN (3x3 conv x 2) + Global Average Pooling
-> Linear projection: 128-dim -> 768-dim (z_freq)
```

Core implementation:

```python
def phase_reconstruct(self, x):
    fft = torch.fft.fft2(x, norm='ortho')
    phase = torch.angle(fft)
    phase_fft = torch.exp(1j * phase)   # normalize magnitude to 1
    phase_img = torch.fft.ifft2(phase_fft, norm='ortho').real
    phase_img = phase_img - phase_img.mean(dim=(-2,-1), keepdim=True)
    phase_img = phase_img / phase_img.std(dim=(-2,-1), keepdim=True).clamp(min=1e-6)
    return phase_img
```

#### Adapter_MoElayer (Phase-Guided Routing)

Integrates phase features into the Adapter-MoE gating mechanism.

```python
x_global = torch.mean(x, dim=1)                    # [B, 768]
gate_input = torch.cat([x_global, z_freq], dim=-1) # [B, 1536]
# -> Noisy Top-k Routing selects experts
```

---

### `dataset.py`

| Class / Variable | Description |
|------------------|-------------|
| `FF_Dataset` | Dataset class for FF++ training and validation |
| `jpeg_aug_transform` | JPEG compression augmentation (quality 30-70, p=0.5) |
| `base_transform` | Base preprocessing (Resize 224, Normalize) |

JPEG augmentation configuration:

```python
alb.ImageCompression(quality_lower=30, quality_upper=70, p=0.5)
```

---

### `train.py`

| Item | Detail |
|------|--------|
| Optimizer | `Adam(lr=3e-5, weight_decay=1e-5)` |
| Scheduler | `StepLR(step_size=5, gamma=0.5)` |
| Loss | `CrossEntropy Loss + MoE Load-Balancing Loss` |
| Logging | TensorBoard (saved under `runs/`) |
| Checkpoints | Saved every epoch; best AUC model saved as `best_model.pkl` |

---

### `inference.py`

Performs video-level prediction by averaging frame-level scores within each video.

| Metric | Description |
|--------|-------------|
| ACC | Video-level accuracy |
| AUC | Area under the ROC curve |
| EER | Equal Error Rate |

---

### `utils.py`

| Function | Description |
|----------|-------------|
| `compute_auc` | Computes AUC score |
| `compute_eer` | Computes Equal Error Rate |
| `compute_apcer_bpcer_acer` | Computes APCER, BPCER, and ACER |

---
