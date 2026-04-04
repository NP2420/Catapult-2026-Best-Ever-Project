"""
train.py — Train the EngagementModel on DAiSEE.

Usage (local test):
    python train.py --epochs 2 --batch_size 8 --max_samples 50

Usage (on Gautschi — via slurm/train.slurm):
    sbatch slurm/train.slurm
"""

import argparse
import math
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


# How many epochs to keep the CNN backbone frozen.
# For the first UNFREEZE_EPOCH epochs only the Transformer + heads train.
# After that the backbone unfreezes with a 10x lower LR to avoid destroying
# the ImageNet features that took millions of images to learn.
UNFREEZE_EPOCH = 10


# Loss: per-dimension weighted MSE

class WeightedDimLoss(nn.Module):
    """
    MSE loss with:
      - label smoothing  : pulls targets toward 0.5, reduces overconfidence on
                           noisy DAiSEE annotations
      - per-dim weights  : upweights confusion and frustration, which appear
                           rarely in DAiSEE and get swamped by boredom/engagement
                           without explicit emphasis

    Dimension order matches LABEL_NAMES: boredom, engagement, confusion, frustration
    """

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        # Weights tuned for DAiSEE label distribution:
        #   boredom     1.5  — more common but still under-predicted
        #   engagement  1.0  — most common, baseline weight
        #   confusion   1.8  — rare, needs emphasis
        #   frustration 2.0  — rarest, highest weight
        self.register_buffer(
            "dim_weights",
            torch.tensor([1.5, 1.0, 1.8, 2.0], dtype=torch.float32)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        # per_dim_mse shape: (4,)  — mean over batch for each dimension
        per_dim_mse = ((pred - smooth_target) ** 2).mean(dim=0)
        return (per_dim_mse * self.dim_weights.to(per_dim_mse.device)).mean()


# LR schedule: linear warmup → cosine decay
# Built relative to a start_epoch so it works correctly after unfreeze rebuild.

def build_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    start_epoch:   int = 0,
) -> LambdaLR:
    """
    Returns a scheduler whose epoch 0 corresponds to start_epoch in the
    overall training run, so cosine decay is calibrated to remaining epochs.
    """
    remaining = total_epochs - start_epoch

    def lr_lambda(local_epoch: int) -> float:
        if local_epoch < warmup_epochs:
            return (local_epoch + 1) / max(1, warmup_epochs)
        progress = (local_epoch - warmup_epochs) / max(1, remaining - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# Backbone freeze / unfreeze helpers

def freeze_backbone(model) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False
    n_frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"  [freeze] backbone frozen  ({n_frozen/1e6:.1f}M params locked)")


def build_optimizer_frozen(model, lr: float, weight_decay: float) -> AdamW:
    """Optimizer that only covers the Transformer + heads (backbone is frozen)."""
    trainable = [p for n, p in model.named_parameters()
                 if "backbone" not in n and p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  [optim]  training {n_trainable/1e6:.1f}M params (backbone excluded)")
    return AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


def unfreeze_backbone(model, optimizer: AdamW, lr: float, weight_decay: float) -> AdamW:
    """
    Unfreeze backbone and rebuild optimizer with two param groups:
      - backbone params  : lr * 0.1  (fine-tune gently, don't destroy ImageNet features)
      - rest of model    : lr        (continue at current rate)
    """
    for param in model.backbone.parameters():
        param.requires_grad = True

    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"  [unfreeze] backbone unfrozen ({n_backbone/1e6:.1f}M params, LR={lr*0.1:.2e})")

    new_optimizer = AdamW(
        [
            {"params": model.backbone.parameters(),                         "lr": lr * 0.1},
            {"params": [p for n, p in model.named_parameters()
                        if "backbone" not in n],                            "lr": lr},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    return new_optimizer


# One training epoch
# grad_accum_steps > 1 accumulates gradients over N micro-batches before
# stepping, giving an effective batch size of batch_size × grad_accum_steps
# without increasing the per-forward-pass tensor size (which caused the
# 32-bit index overflow at batch_size=128).

def train_epoch(
    model, loader, optimizer, criterion, scaler, device, writer,
    global_step, grad_accum_steps: int = 1,
):
    model.train()
    total_loss = 0.0
    dim_mae    = np.zeros(len(LABEL_NAMES))
    n_batches  = 0

    optimizer.zero_grad()

    for step, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)   # (B, T, C, H, W)
        labels = labels.to(device, non_blocking=True)   # (B, 4)

        with torch.cuda.amp.autocast(enabled=CFG.train.mixed_precision):
            preds = model(frames)
            # Divide loss by accum steps so gradients are averaged correctly
            # across the effective batch before the optimizer step.
            loss  = criterion(preds, labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Un-scale for logging so reported loss matches the original scale
            loss_for_log = loss.item() * grad_accum_steps
            total_loss  += loss_for_log
            dim_mae     += (preds.detach().cpu() - labels.cpu()).abs().mean(dim=0).numpy()
            n_batches   += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/batch_loss", loss_for_log, global_step)

            global_step += 1

    # Flush any leftover accumulated gradients at the end of the epoch
    # (happens when len(loader) is not divisible by grad_accum_steps)
    remainder = len(loader) % grad_accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1), dim_mae / max(n_batches, 1), global_step


# Validation pass

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


# Checkpoint helpers

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  [ckpt] loaded <- {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,   default=60)
    # FIX: default lowered from 128 → 32 to avoid 32-bit index overflow in
    # EfficientNet-B4 when the batch is flattened to (B×T, C, H, W).
    # Use --grad_accum_steps 4 (the new default) for an effective batch of 128.
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--grad_accum_steps", type=int,   default=4,
                        help="Accumulate gradients over N steps; "
                             "effective_bs = batch_size × grad_accum_steps "
                             "(default 4 → effective 128). Set to 1 to disable.")
    parser.add_argument("--lr",               type=float, default=CFG.train.lr)
    parser.add_argument("--num_workers",      type=int,   default=CFG.train.num_workers)
    parser.add_argument("--unfreeze_epoch",   type=int,   default=UNFREEZE_EPOCH,
                        help="Epoch at which to unfreeze the CNN backbone")
    parser.add_argument("--max_samples",      type=int,   default=None,
                        help="Cap training set size (debug mode)")
    parser.add_argument("--resume",           type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--run_name",         type=str,   default="run_02_weighted")
    args = parser.parse_args()

    effective_bs = args.batch_size * args.grad_accum_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Batch size    : {args.batch_size}  (per step)")
    print(f"Accum steps   : {args.grad_accum_steps}")
    print(f"Effective BS  : {effective_bs}")

    # Data
    print("\nBuilding dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # Model
    model = build_model(device=str(device))

    # Freeze backbone — only Transformer + heads train for first N epochs
    freeze_backbone(model)
    backbone_unfrozen = False

    # Optimizer — initially excludes frozen backbone
    optimizer = build_optimizer_frozen(model, args.lr, CFG.train.weight_decay)
    scheduler = build_scheduler(optimizer, CFG.train.warmup_epochs, args.epochs)
    criterion = WeightedDimLoss(smoothing=CFG.train.label_smooth).to(device)
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG.train.mixed_precision)

    print(f"\nLoss: WeightedDimLoss  weights={criterion.dim_weights.tolist()}")
    print(f"Backbone unfreeze at epoch {args.unfreeze_epoch}")

    # Optional resume
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )
        start_epoch += 1
        # If resuming past the unfreeze point, unfreeze immediately
        if start_epoch > args.unfreeze_epoch:
            optimizer = unfreeze_backbone(model, optimizer, args.lr, CFG.train.weight_decay)
            scheduler = build_scheduler(
                optimizer, 0, args.epochs, start_epoch=start_epoch
            )
            backbone_unfrozen = True

    # Logging
    log_dir = LOGS_DIR / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer      = SummaryWriter(log_dir=str(log_dir))
    global_step = start_epoch * len(train_loader)

    print(f"\nTensorBoard logs -> {log_dir}")
    print(f"Checkpoints     -> {CKPT_DIR}\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Unfreeze backbone at the scheduled epoch
        if not backbone_unfrozen and epoch == args.unfreeze_epoch:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}: UNFREEZING backbone with LR={args.lr*0.1:.2e}")
            print(f"{'='*60}\n")
            optimizer = unfreeze_backbone(model, optimizer, args.lr, CFG.train.weight_decay)
            # Rebuild scheduler for remaining epochs, no warmup (already warmed up)
            scheduler = build_scheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=args.epochs,
                start_epoch=epoch,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=CFG.train.mixed_precision)
            backbone_unfrozen = True

        train_loss, train_mae, global_step = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, writer, global_step,
            grad_accum_steps=args.grad_accum_steps,
        )
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        # Get backbone LR if unfrozen, otherwise show head LR
        if backbone_unfrozen and len(optimizer.param_groups) > 1:
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_heads    = optimizer.param_groups[1]["lr"]
            lr_display  = f"lr_bb={lr_backbone:.2e} lr_hd={lr_heads:.2e}"
        else:
            lr_display = f"lr={optimizer.param_groups[0]['lr']:.2e}"

        # Console output
        frozen_tag = "" if backbone_unfrozen else " [backbone frozen]"
        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"{lr_display}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"({elapsed:.0f}s){frozen_tag}"
        )
        for i, name in enumerate(LABEL_NAMES):
            print(f"  {name:<12}  train_MAE={train_mae[i]:.4f}  val_MAE={val_mae[i]:.4f}")

        # TensorBoard
        writer.add_scalar("loss/train",           train_loss,             epoch)
        writer.add_scalar("loss/val",             val_loss,               epoch)
        writer.add_scalar("backbone_unfrozen",    int(backbone_unfrozen), epoch)
        for i, name in enumerate(LABEL_NAMES):
            writer.add_scalar(f"mae_train/{name}", train_mae[i], epoch)
            writer.add_scalar(f"mae_val/{name}",   val_mae[i],   epoch)
        if backbone_unfrozen and len(optimizer.param_groups) > 1:
            writer.add_scalar("lr/backbone", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("lr/heads",    optimizer.param_groups[1]["lr"], epoch)
        else:
            writer.add_scalar("lr/heads", optimizer.param_groups[0]["lr"], epoch)

        # Checkpointing
        ckpt_state = {
            "epoch":             epoch,
            "model":             model.state_dict(),
            "optimizer":         optimizer.state_dict(),
            "scheduler":         scheduler.state_dict(),
            "val_loss":          val_loss,
            "best_val_loss":     best_val_loss,
            "backbone_unfrozen": backbone_unfrozen,
            "config":            CFG,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_state, CKPT_DIR / "best.pt")
            print("  *** New best checkpoint ***")

        if (epoch + 1) % CFG.train.save_every == 0:
            save_checkpoint(ckpt_state, CKPT_DIR / f"epoch_{epoch+1:03d}.pt")

    save_checkpoint(ckpt_state, CKPT_DIR / "final.pt")
    writer.close()

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best ckpt     : {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()