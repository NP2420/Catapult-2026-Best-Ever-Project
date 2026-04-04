"""
train.py — Train the EngagementModel on DAiSEE.

Features:
  - Mixed-precision training (torch.cuda.amp) for ~30% speedup on H100
  - Linear warmup + cosine annealing LR schedule
  - Label smoothing on targets to reduce overconfidence
  - Gradient clipping for training stability
  - TensorBoard logging of losses, per-dimension MAE, and LR
  - Automatic checkpointing (best val loss + every N epochs)
  - Resume from checkpoint via --resume flag

Usage (local test):
    python train.py --epochs 2 --batch_size 8 --max_samples 50

Usage (on Gautschi — via slurm/train.slurm):
    sbatch slurm/train.slurm
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from config import CFG, CKPT_DIR, LOGS_DIR, LABEL_NAMES
from dataset import build_dataloaders
from model import build_model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SmoothedMSELoss(nn.Module):
    """
    MSE loss with label smoothing.

    Smoothing pulls targets toward 0.5, which:
      - prevents the model becoming overconfident on noisy DAiSEE labels
      - acts as a weak regulariser on the output range
    """

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.mse_loss(pred, smooth_target)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs             # linear warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))   # cosine decay

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, optimizer, criterion, scaler, device, writer, global_step
):
    model.train()
    total_loss = 0.0
    dim_mae    = np.zeros(len(LABEL_NAMES))
    n_batches  = 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)   # (B, T, C, H, W)
        labels = labels.to(device, non_blocking=True)   # (B, 4)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=CFG.train.mixed_precision):
            preds = model(frames)                        # (B, 4)
            loss  = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics (detach to avoid keeping graph)
        total_loss += loss.item()
        dim_mae    += (preds.detach().cpu() - labels.cpu()).abs().mean(dim=0).numpy()
        n_batches  += 1

        if global_step % 50 == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)

        global_step += 1

    return total_loss / n_batches, dim_mae / n_batches, global_step


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dim_mae    = np.zeros(len(LABEL_NAMES))
    n_batches  = 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=CFG.train.mixed_precision):
            preds = model(frames)
            loss  = criterion(preds, labels)

        total_loss += loss.item()
        dim_mae    += (preds.cpu() - labels.cpu()).abs().mean(dim=0).numpy()
        n_batches  += 1

    return total_loss / n_batches, dim_mae / n_batches


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer  and "optimizer"  in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  and "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  [ckpt] loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=CFG.train.epochs)
    parser.add_argument("--batch_size",  type=int,   default=CFG.train.batch_size)
    parser.add_argument("--lr",          type=float, default=CFG.train.lr)
    parser.add_argument("--num_workers", type=int,   default=CFG.train.num_workers)
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Cap training set size (debug mode)")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--run_name",    type=str,   default="run_01")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -------------------------------------------------------------------
    # Data
    print("\nBuilding dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # -------------------------------------------------------------------
    # Model, optimizer, scheduler, loss
    model     = build_model(device=str(device))
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=CFG.train.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler  = build_scheduler(optimizer, CFG.train.warmup_epochs, args.epochs)
    criterion  = SmoothedMSELoss(CFG.train.label_smooth)
    scaler     = torch.cuda.amp.GradScaler(enabled=CFG.train.mixed_precision)

    # -------------------------------------------------------------------
    # Optional resume
    start_epoch    = 0
    best_val_loss  = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )
        start_epoch += 1

    # -------------------------------------------------------------------
    # Logging
    log_dir = LOGS_DIR / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    global_step = start_epoch * len(train_loader)

    print(f"\nTensorBoard logs → {log_dir}")
    print(f"Checkpoints     → {CKPT_DIR}\n")

    # -------------------------------------------------------------------
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_mae, global_step = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, writer, global_step
        )
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # -------------------------------------------------------------------
        # Console output
        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"lr={lr_now:.2e}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )
        for i, name in enumerate(LABEL_NAMES):
            print(
                f"  {name:<12}  train_MAE={train_mae[i]:.4f}  val_MAE={val_mae[i]:.4f}"
            )

        # -------------------------------------------------------------------
        # TensorBoard
        writer.add_scalar("loss/train",     train_loss,    epoch)
        writer.add_scalar("loss/val",       val_loss,      epoch)
        writer.add_scalar("lr",             lr_now,        epoch)
        for i, name in enumerate(LABEL_NAMES):
            writer.add_scalar(f"mae_train/{name}", train_mae[i], epoch)
            writer.add_scalar(f"mae_val/{name}",   val_mae[i],   epoch)

        # -------------------------------------------------------------------
        # Checkpointing
        ckpt_state = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "config":        CFG,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_state, CKPT_DIR / "best.pt")
            print("  *** New best checkpoint ***")

        if (epoch + 1) % CFG.train.save_every == 0:
            save_checkpoint(ckpt_state, CKPT_DIR / f"epoch_{epoch+1:03d}.pt")

    # -------------------------------------------------------------------
    # Save final checkpoint
    save_checkpoint(ckpt_state, CKPT_DIR / "final.pt")
    writer.close()

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best ckpt     : {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
