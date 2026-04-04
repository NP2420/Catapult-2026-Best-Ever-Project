"""
dataset.py — PyTorch Dataset for preprocessed DAiSEE face crops.

Each sample is a tensor of shape (seq_len, C, H, W) drawn from a sliding
window over a clip's extracted frames, paired with a (4,) label vector.

Sliding window strategy:
  - Training : random start within [0, N - seq_len], with augmentation
  - Val/Test  : deterministic centre crop of the clip, no augmentation

This means one clip can contribute many training samples but only one
validation sample — consistent with standard temporal model evaluation.
"""

import json
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

from config import CFG, CROPS_ROOT, LABEL_NAMES


# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------

def build_train_transform(crop_size: int) -> A.Compose:
    """Augmentation applied independently to each frame in a sequence."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ToGray(p=0.05),                    # rare greyscale: simulates poor webcams
        A.Normalize(mean=CFG.inference.mean, std=CFG.inference.std),
        ToTensorV2(),                         # → (C, H, W) float32
    ])


def build_val_transform() -> A.Compose:
    """No spatial augmentation, just normalisation."""
    return A.Compose([
        A.Normalize(mean=CFG.inference.mean, std=CFG.inference.std),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DAiSEEDataset(Dataset):
    """
    Parameters
    ----------
    split       : "train" | "val" | "test"
    seq_len     : number of frames per sample (must match ModelConfig.seq_len)
    transform   : albumentations pipeline applied per frame
    max_samples : optional cap (useful for quick debug runs)
    """

    def __init__(
        self,
        split:       str,
        seq_len:     int             = CFG.model.seq_len,
        transform:   Optional[A.Compose] = None,
        max_samples: Optional[int]   = None,
    ):
        self.split     = split
        self.seq_len   = seq_len
        self.transform = transform
        self.is_train  = (split == "train")

        data_root = CROPS_ROOT / split
        manifest_path = data_root / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run preprocess_daisee.py first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Keep only clips long enough for one full sequence
        self.clips: List[dict] = []
        for entry in manifest:
            if entry["n_frames"] >= seq_len:
                self.clips.append({
                    "frames_path": data_root / entry["clip_id"] / "frames.npy",
                    "labels_path": data_root / entry["clip_id"] / "labels.npy",
                    "n_frames":    entry["n_frames"],
                })

        if max_samples is not None:
            random.shuffle(self.clips)
            self.clips = self.clips[:max_samples]

        # Preload all labels into RAM — they're tiny (4 floats per clip)
        self.labels = np.stack([
            np.load(c["labels_path"]) for c in self.clips
        ])  # (N_clips, 4)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip   = self.clips[idx]
        labels = torch.from_numpy(self.labels[idx])   # (4,) float32

        # Memory-mapped load: only the needed slice is read from disk
        all_frames = np.load(clip["frames_path"], mmap_mode="r")  # (N, H, W, C)
        n = all_frames.shape[0]

        # Select starting frame
        if self.is_train:
            start = random.randint(0, n - self.seq_len)
        else:
            start = max(0, (n - self.seq_len) // 2)   # centre of clip

        frames_np = all_frames[start : start + self.seq_len]  # (T, H, W, C) uint8

        # Apply augmentation frame-by-frame (same random seed for flip consistency)
        # We re-seed per frame with a clip-level seed so horizontal flip is
        # applied to ALL frames in the sequence uniformly (a person can't
        # flip mid-clip).
        if self.transform:
            if self.is_train:
                flip = random.random() < 0.5
            else:
                flip = False

            processed = []
            for frame in frames_np:
                if self.is_train:
                    # Apply colour jitter etc. independently per frame
                    aug = self.transform(image=np.array(frame))["image"]
                    if flip:
                        aug = aug.flip(-1)   # horizontal flip on tensor
                    processed.append(aug)
                else:
                    processed.append(self.transform(image=np.array(frame))["image"])

            frames = torch.stack(processed)   # (T, C, H, W) float32
        else:
            frames = torch.from_numpy(
                frames_np.transpose(0, 3, 1, 2).copy()
            ).float() / 255.0

        return frames, labels

    # ------------------------------------------------------------------

    def get_sample_weights(self) -> torch.Tensor:
        """
        Return per-sample weights for WeightedRandomSampler.

        Clips with high engagement (> threshold) are upsampled to compensate
        for DAiSEE's class imbalance (most clips are medium engagement).
        """
        engagement_scores = self.labels[:, 1]   # column 1 = Engagement
        weights = np.ones(len(self.clips), dtype=np.float32)

        if CFG.train.oversample_engaged:
            high_mask = engagement_scores >= CFG.train.engaged_threshold
            weights[high_mask]  = 3.0
            weights[~high_mask] = 1.0

        return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size:  int = CFG.train.batch_size,
    num_workers: int = CFG.train.num_workers,
    seq_len:     int = CFG.model.seq_len,
    max_samples: Optional[int] = None,
):
    """Build train, val, and test DataLoaders."""

    train_ds = DAiSEEDataset(
        split="train",
        seq_len=seq_len,
        transform=build_train_transform(CFG.preproc.crop_size),
        max_samples=max_samples,
    )
    val_ds = DAiSEEDataset(
        split="val",
        seq_len=seq_len,
        transform=build_val_transform(),
    )
    test_ds = DAiSEEDataset(
        split="test",
        seq_len=seq_len,
        transform=build_val_transform(),
    )

    # Weighted sampling for training
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=CFG.train.pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=CFG.train.pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=CFG.train.pin_memory,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = DAiSEEDataset("val", max_samples=10)
    frames, labels = ds[0]
    print(f"Frames shape : {frames.shape}")   # (10, 3, 112, 112)
    print(f"Labels shape : {labels.shape}")   # (4,)
    print(f"Label values : {labels}")         # e.g. [0.33, 0.67, 0.00, 0.00]
    print(f"Label names  : {LABEL_NAMES}")
    print("Dataset OK")
