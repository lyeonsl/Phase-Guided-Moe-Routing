"""
dataset.py
Dataset classes for FaceForensics++ deepfake detection.

Folder structure expected:
    <split>/real/<video_id>/<frame>.png  -> label 0
    <split>/fake/<video_id>/<frame>.png  -> label 1
"""
import cv2
import os
import torch
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

img_size = 224
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

base_transform = alb.Compose([
    alb.Resize(img_size, img_size),
    alb.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

jpeg_aug_transform = alb.Compose([
    alb.Resize(img_size, img_size),
    alb.ImageCompression(quality_lower=30, quality_upper=70, p=0.5),
    alb.Normalize(mean=mean, std=std),
    ToTensorV2(),
])


class FFPP_Dataset(Dataset):
    """Training dataset with optional JPEG compression augmentation."""

    def __init__(self, path, frame=20, phase='train', jpeg_aug=False):
        super().__init__()
        assert phase in ['train', 'valid', 'test']
        self.path = path
        self.frame = frame
        self.phase = phase
        self.transform = jpeg_aug_transform if (jpeg_aug and phase == 'train') else base_transform
        self.list = self._generate_list()
        self.images = [line.strip().split()[0] for line in self.list]
        self.labels = [line.strip().split()[1] for line in self.list]

    def _generate_list(self):
        list_ = []

        # real (label=0) — 4x sampling in train to balance against 4 fake types
        real_path = os.path.join(self.path, 'real')
        if os.path.exists(real_path):
            frame = 4 * self.frame if self.phase == 'train' else self.frame
            for v in sorted(os.listdir(real_path)):
                v_path = os.path.join(real_path, v)
                if not os.path.isdir(v_path):
                    continue
                pic = sorted([p for p in os.listdir(v_path) if p.lower().endswith(('.png', '.jpg', '.jpeg'))],
                             key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
                if not pic:
                    continue
                if len(pic) < frame:
                    pic = pic * (frame // len(pic) + 1)
                interval = len(pic) // frame
                list_ += [os.path.join(v_path, pic[i * interval]) + ' 0\n' for i in range(frame)]

        # fake (label=1)
        fake_path = os.path.join(self.path, 'fake')
        if os.path.exists(fake_path):
            for v in sorted(os.listdir(fake_path)):
                v_path = os.path.join(fake_path, v)
                if not os.path.isdir(v_path):
                    continue
                pic = sorted([p for p in os.listdir(v_path) if p.lower().endswith(('.png', '.jpg', '.jpeg'))],
                             key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
                if not pic:
                    continue
                if len(pic) < self.frame:
                    pic = pic * (self.frame // len(pic) + 1)
                interval = len(pic) // self.frame
                list_ += [os.path.join(v_path, pic[i * interval]) + ' 1\n' for i in range(self.frame)]

        print(f'[{self.phase}] total {len(list_)} frames loaded')
        return list_

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        fn, label = self.images[item], self.labels[item]
        img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img, int(label)


class TestDataset(Dataset):
    """Evaluation dataset — returns all frames per video as a single batch."""

    def __init__(self, path, dataset='FFPP', frame=20):
        super().__init__()
        self.frame = frame
        self.list = self._gen_video_list(path)

    def _gen_video_list(self, path):
        list_ = []
        for label_name, label_id in [('real', 0), ('fake', 1)]:
            label_path = os.path.join(path, label_name)
            if os.path.exists(label_path):
                for v in sorted(os.listdir(label_path)):
                    v_path = os.path.join(label_path, v)
                    if os.path.isdir(v_path):
                        list_.append(f'{v_path} {label_id}\n')
        print(f'[test] total {len(list_)} videos loaded')
        return list_

    def _load_image(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return base_transform(image=img)['image'].unsqueeze(0)

    def _load_video(self, path):
        pic = sorted([p for p in os.listdir(path) if p.lower().endswith(('.png', '.jpg', '.jpeg'))],
                     key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
        if not pic:
            return torch.zeros(self.frame, 3, img_size, img_size)
        if len(pic) < self.frame:
            pic = pic * (self.frame // len(pic) + 1)
        interval = len(pic) // self.frame
        frames = [self._load_image(os.path.join(path, pic[i * interval])) for i in range(self.frame)]
        return torch.cat(frames, dim=0)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        v_path, v_label = self.list[item].strip().split()
        return self._load_video(v_path), int(v_label)
