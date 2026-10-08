"""
inference.py
Inference script for MoE-FFD deepfake detection models.
Evaluates ACC, AUC, EER on a given dataset split.

Usage:
    python inference.py \
        --data_root /path/to/dataset/c40_crop \
        --ckpt_path /path/to/model.pkl \
        --model moeffd_phase \
        --device cuda:0
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class FlatImageDataset(Dataset):
    """
    Loads frames from a split with the following structure:
        <data_root>/real/<video_id>/<frame>.png   -> label 0
        <data_root>/fake/<video_id>/<frame>.png   -> label 1
    """
    def __init__(self, real_root, fake_root, transform, max_frames=32):
        self.samples = []
        self.transform = transform

        for vid in sorted(os.listdir(real_root)):
            vid_dir = os.path.join(real_root, vid)
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted(glob(os.path.join(vid_dir, '*.png')) +
                            glob(os.path.join(vid_dir, '*.jpg')))[:max_frames]
            for f in frames:
                self.samples.append((f, 0))

        for entry in sorted(os.listdir(fake_root)):
            entry_dir = os.path.join(fake_root, entry)
            if not os.path.isdir(entry_dir):
                continue
            # Support both flat (fake/video_id/) and nested (fake/manip_type/video_id/)
            sub_dirs = [d for d in os.listdir(entry_dir) if os.path.isdir(os.path.join(entry_dir, d))]
            if sub_dirs:
                for vid in sorted(sub_dirs):
                    vid_dir = os.path.join(entry_dir, vid)
                    frames = sorted(glob(os.path.join(vid_dir, '*.png')) +
                                    glob(os.path.join(vid_dir, '*.jpg')))[:max_frames]
                    for f in frames:
                        self.samples.append((f, 1))
            else:
                frames = sorted(glob(os.path.join(entry_dir, '*.png')) +
                                glob(os.path.join(entry_dir, '*.jpg')))[:max_frames]
                for f in frames:
                    self.samples.append((f, 1))

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f'  {len(self.samples)} frames total (real: {n_real}, fake: {n_fake})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(labels, probs):
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = (probs >= 0.5).astype(int)
    acc    = (preds == labels).mean() * 100
    try:
        auc_score = metrics.roc_auc_score(labels, probs)
        fpr, tpr, _ = metrics.roc_curve(labels, probs)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        auc_score, eer = 0.0, 0.5
    return {'ACC': round(acc, 4), 'AUC': round(auc_score, 4), 'EER': round(eer * 100, 4)}


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def run_inference(data_root, ckpt_path, model_type, device,
                  uncertainty_routing=False, entropy_thresh=1.0, max_frames=32):
    """
    Args:
        data_root   : path containing real/ and fake/ directories
        ckpt_path   : path to model checkpoint (.pkl or .tar)
        model_type  : 'moeffd' | 'moeffd_phase'
        device      : torch device string, e.g. 'cuda:0'
    """
    print(f'\n[{model_type}] checkpoint: {ckpt_path}')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if model_type == 'moeffd_phase':
        from ViT_MoE_phase_aug import vit_base_patch16_224_in21k
        model = vit_base_patch16_224_in21k(
            pretrained=False, num_classes=2,
            use_freq_module=True,
            uncertainty_routing=uncertainty_routing,
            entropy_thresh=entropy_thresh,
        ).to(device)
    else:  # baseline
        from ViT_MoE_phase_aug import vit_base_patch16_224_in21k
        model = vit_base_patch16_224_in21k(
            pretrained=False, num_classes=2,
            use_freq_module=False,
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = FlatImageDataset(
        os.path.join(data_root, 'real'),
        os.path.join(data_root, 'fake'),
        transform, max_frames=max_frames)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=model_type, ncols=70):
            imgs = imgs.to(device)
            out, _, _, _, _, _ = model(imgs)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(prob.tolist())
            all_labels.extend(labels.numpy().tolist())

    return compute_metrics(all_labels, all_probs)


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def save_plots(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if not results:
        return

    models = list(results.keys())
    accs   = [results[m]['ACC']       for m in models]
    aucs   = [results[m]['AUC'] * 100 for m in models]
    eers   = [results[m]['EER']       for m in models]
    colors = ['#185FA5', '#0F6E56', '#E24B4A', '#F5A623'][:len(models)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Deepfake Detection Results', fontsize=13, fontweight='bold')
    for ax, vals, title in zip(axes, [accs, aucs, eers], ['Accuracy (%)', 'AUC (%)', 'EER (%) ↓']):
        bars = ax.bar(models, vals, color=colors, width=0.5, edgecolor='white', linewidth=1.2)
        ax.set_title(title, fontsize=12)
        ylim_top = max(vals) * 1.15 if max(vals) > 0 else 1
        ax.set_ylim(0, ylim_top)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + ylim_top * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nChart saved: {out_path}')

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root containing real/ and fake/ subdirs')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint (.pkl or .tar)')
    parser.add_argument('--model', type=str, default='moeffd_phase',
                        choices=['moeffd', 'moeffd_phase'],
                        help='Model type to evaluate')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--uncertainty_routing', action='store_true', default=False)
    parser.add_argument('--entropy_thresh', type=float, default=1.0)
    args = parser.parse_args()

    result = run_inference(
        data_root=args.data_root,
        ckpt_path=args.ckpt_path,
        model_type=args.model,
        device=args.device,
        uncertainty_routing=args.uncertainty_routing,
        entropy_thresh=args.entropy_thresh,
        max_frames=args.max_frames,
    )

    print(f'\n{"="*50}')
    print(f'Model : {args.model}')
    print(f'ACC   : {result["ACC"]:.2f}%')
    print(f'AUC   : {result["AUC"]:.4f}')
    print(f'EER   : {result["EER"]:.2f}%')
    print(f'{"="*50}')

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({args.model: result}, f, indent=2)
