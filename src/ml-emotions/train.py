"""
train.py
--------
Train a YOLOv8x-cls emotion classifier on AffectNet using all 8 H100 GPUs.
Designed for Gautschi -p ai partition.

Usage (via SLURM, see slurm/train.sh):
    python train.py [options]

Direct run (single GPU test):
    python train.py --gpus 0 --epochs 5 --batch 64
"""

import argparse
import os
import time
from pathlib import Path

from ultralytics import YOLO
import torch


# ─── Emotion classes (must match folder names after prepare_data.py) ─────────
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 emotion classifier")
    p.add_argument("--data_dir",    type=str,   default="./data",
                   help="Path to data/ containing train/ and val/")
    p.add_argument("--model",       type=str,   default="yolov8x-cls.pt",
                   help="YOLO model variant (yolov8n/s/m/l/x-cls.pt)")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--imgsz",       type=int,   default=224,
                   help="Input image size (224 or 320 for better accuracy)")
    p.add_argument("--batch",       type=int,   default=-1,
                   help="Batch size per GPU. -1 = auto-scale for H100 80 GB")
    p.add_argument("--workers",     type=int,   default=14,
                   help="DataLoader workers per GPU (14 cores/GPU on Gautschi)")
    p.add_argument("--gpus",        type=str,   default="0,1,2,3,4,5,6,7",
                   help="Comma-separated GPU IDs, e.g. '0,1,2,3,4,5,6,7'")
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early stopping patience (epochs)")
    p.add_argument("--name",        type=str,   default="emotion_yolov8x",
                   help="Run name under outputs/")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dropout",     type=float, default=0.2)
    return p.parse_args()


def auto_batch(imgsz: int) -> int:
    """
    Estimate a good batch size per GPU for H100 80 GB.
    Rule of thumb: ~1.5–2 GB per image at imgsz=224 for yolov8x-cls.
    """
    # Each H100 has 80 GB; leave 10 GB headroom
    available_gb = 70
    # approx MB per sample at imgsz=224 with fp16
    mb_per_sample = 0.8
    batch = int((available_gb * 1024) / mb_per_sample)
    # Round to nearest power of 2
    batch = 2 ** (batch - 1).bit_length()
    batch = min(batch, 512)  # practical ceiling
    print(f"[auto_batch] Estimated batch size per GPU: {batch}")
    return batch


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8 Emotion Classifier — Training")
    print("=" * 60)
    print(f"  Model     : {args.model}")
    print(f"  GPUs      : {args.gpus}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Data dir  : {args.data_dir}")

    # Validate data directory
    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    for split in ["train", "val"]:
        for emotion in EMOTIONS:
            p = data_dir / split / emotion
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}. Run prepare_data.py first."
                )

    # Build device string
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    device = ",".join(str(g) for g in gpu_ids) if num_gpus > 1 else gpu_ids[0]

    print(f"  Device    : {device}  ({num_gpus} GPU(s))")
    print(f"  CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in gpu_ids:
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {name}  ({mem:.0f} GB)")

    # Resolve batch size
    batch = args.batch if args.batch > 0 else auto_batch(args.imgsz)
    # For multi-GPU, YOLO multiplies batch × num_gpus internally
    print(f"  Batch/GPU : {batch}   (total effective: {batch * num_gpus})")

    # Load or resume model
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    t0 = time.time()

    results = model.train(
        data=str(data_dir),       # YOLOv8 cls: data = folder with train/ val/
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers * num_gpus,
        device=device,
        amp=True,                  # Mixed precision (BF16 on H100)
        optimizer="AdamW",
        lr0=args.lr,
        lrf=0.01,                  # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        # Augmentation (strong for better generalisation)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,                # Mosaic is for detection, keep off for cls
        erasing=0.4,               # Random erasing — helps with occlusion
        auto_augment="randaugment",
        # Output
        project="outputs",
        name=args.name,
        save=True,
        save_period=10,            # Save checkpoint every N epochs
        plots=True,
        verbose=True,
        seed=42,
        deterministic=False,       # Non-deterministic is faster on H100
        cache=True,                # Cache images in RAM (1 TB available!)
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.2f} h")
    print(f"Best model  : outputs/{args.name}/weights/best.pt")
    print(f"Last model  : outputs/{args.name}/weights/last.pt")
    print(f"Top-1 acc   : {results.results_dict.get('metrics/accuracy_top1', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
