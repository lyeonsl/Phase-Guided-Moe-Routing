# Phase-Guided-Moe-Routing
2026 Sookmyung Women's University Graduation Project

A ViT-B/16 based deepfake detection model combining LoRA MoE, Conv2d-Diff Adapter MoE, and a PhaseModule for frequency-aware expert routing.

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
data/FF++/c23_crop/
├── train/
│   ├── real/<video_id>/<frame>.png
│   └── fake/<video_id>/<frame>.png
└── val/
    ├── real/<video_id>/<frame>.png
    └── fake/<video_id>/<frame>.png
```

## Training

```bash
python train.py \
    --train_path data/FF++/c23_crop/train \
    --valid_path data/FF++/c23_crop/val \
    --model_dir models/train/ \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --device 0
```

Optional flags:
- `--jpeg_aug` : enable JPEG compression augmentation (for c40 robustness)
- `--uncertainty_routing` : enable entropy-based adaptive top-k routing
- `--entropy_thresh 1.0` : entropy threshold for uncertainty routing

## Inference

```bash
python inference.py \
    --data_root /path/to/dataset/c40_crop \
    --ckpt_path /path/to/model.pkl \
    --model moeffd_phase \
    --device cuda:0
```

Outputs `results/results.json` with ACC, AUC, EER.

## Files

| File | Description |
|------|-------------|
| `ViT_MoE_phase_aug.py` | Model definition (ViT + LoRA MoE + Adapter MoE + PhaseModule) |
| `train.py` | Training script |
| `dataset.py` | Dataset classes for FF++ |
| `utils.py` | Evaluation metrics (ACC, AUC, EER, APCER, BPCER, ACER) |
| `inference.py` | Inference script |
