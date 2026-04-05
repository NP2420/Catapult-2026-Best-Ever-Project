"""
validate.py
-----------
Evaluate the trained emotion model on the val set.
Outputs: accuracy, per-class precision/recall/F1, confusion matrix, and
         a misclassified-samples gallery.

Usage:
    python validate.py --model outputs/emotion_yolov8x/weights/best.pt
    python validate.py --model outputs/emotion_yolov8x/weights/best.pt --split train
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO


EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_COLORS = {
    "anger":    "#E74C3C",
    "contempt": "#9B59B6",
    "disgust":  "#27AE60",
    "fear":     "#F39C12",
    "happy":    "#F1C40F",
    "neutral":  "#95A5A6",
    "sad":      "#2980B9",
    "surprise": "#1ABC9C",
}


def parse_args():
    p = argparse.ArgumentParser(description="Validate emotion classifier")
    p.add_argument("--model",    type=str, required=True,
                   help="Path to best.pt or last.pt")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--split",    type=str, default="val",
                   choices=["val", "train"])
    p.add_argument("--imgsz",    type=int, default=224)
    p.add_argument("--batch",    type=int, default=256)
    p.add_argument("--device",   type=str, default="0")
    p.add_argument("--out_dir",  type=str, default="./outputs/validation",
                   help="Where to save plots and reports")
    p.add_argument("--top_k",    type=int, default=20,
                   help="Number of misclassified samples to save per class")
    return p.parse_args()


def collect_images(split_dir: Path):
    """Return (paths, labels) lists where label is the class index."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, labels = [], []
    for idx, emotion in enumerate(EMOTIONS):
        emo_dir = split_dir / emotion
        if not emo_dir.exists():
            print(f"  [WARN] Missing folder: {emo_dir}")
            continue
        imgs = [f for f in emo_dir.iterdir() if f.suffix.lower() in exts]
        paths.extend(imgs)
        labels.extend([idx] * len(imgs))
    return paths, labels


def run_inference(model: YOLO, image_paths: list, imgsz: int, batch: int, device: str):
    """Run batched inference and return list of predicted class indices."""
    preds = []
    topk_probs = []

    for i in tqdm(range(0, len(image_paths), batch), desc="  Inferring"):
        batch_paths = [str(p) for p in image_paths[i : i + batch]]
        results = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            device=device,
            verbose=False,
            stream=False,
        )
        for r in results:
            probs = r.probs
            preds.append(int(probs.top1))
            topk_probs.append(probs.data.cpu().numpy())

    return preds, np.stack(topk_probs)


def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
    disp.plot(ax=axes[0], colorbar=False, xticks_rotation=45)
    axes[0].set_title("Confusion Matrix (counts)", fontsize=14, fontweight="bold")

    # Normalized
    disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=EMOTIONS)
    disp_n.plot(ax=axes[1], colorbar=False, xticks_rotation=45, values_format=".2f")
    axes[1].set_title("Confusion Matrix (normalized)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class_accuracy(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [EMOTION_COLORS[e] for e in EMOTIONS]
    bars = ax.bar(EMOTIONS, per_class_acc * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=np.mean(per_class_acc) * 100, color="red", linestyle="--",
               label=f"Mean: {np.mean(per_class_acc)*100:.1f}%")
    ax.legend(fontsize=11)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_misclassified(image_paths, y_true, y_pred, out_dir: Path, top_k: int):
    """Save a gallery of misclassified images for error analysis."""
    gallery_dir = out_dir / "misclassified"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    misclassified = [
        (image_paths[i], y_true[i], y_pred[i])
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]

    # Group by true class
    from collections import defaultdict
    by_true = defaultdict(list)
    for path, true, pred in misclassified:
        by_true[true].append((path, pred))

    for true_idx, samples in by_true.items():
        samples = samples[:top_k]
        n = len(samples)
        if n == 0:
            continue
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        axes = np.array(axes).flatten()
        fig.suptitle(f"True: {EMOTIONS[true_idx]}  (mis-classified)",
                     fontsize=13, fontweight="bold", color=EMOTION_COLORS[EMOTIONS[true_idx]])

        for ax, (path, pred_idx) in zip(axes, samples):
            try:
                img = Image.open(path).convert("RGB").resize((224, 224))
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "?", ha="center", va="center")
            ax.set_title(f"Pred: {EMOTIONS[pred_idx]}", fontsize=8,
                         color=EMOTION_COLORS[EMOTIONS[pred_idx]])
            ax.axis("off")

        for ax in axes[len(samples):]:
            ax.axis("off")

        out = gallery_dir / f"misclassified_{EMOTIONS[true_idx]}.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()

    print(f"  Misclassified galleries → {gallery_dir}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Validation  |  model: {args.model}")
    print(f"             split: {args.split}  |  device: {args.device}")
    print("=" * 60)

    # Load model
    model = YOLO(args.model)

    # Collect images
    split_dir = Path(args.data_dir) / args.split
    image_paths, y_true = collect_images(split_dir)
    print(f"\nTotal images: {len(image_paths)}")

    # Inference
    print("\nRunning inference...")
    y_pred, probs = run_inference(
        model, image_paths, args.imgsz, args.batch, args.device
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    overall_acc = (y_true_arr == y_pred_arr).mean()

    # Top-5 accuracy
    top5_correct = sum(
        y_true[i] in np.argsort(probs[i])[-5:]
        for i in range(len(y_true))
    )
    top5_acc = top5_correct / len(y_true)

    print(f"\n{'='*40}")
    print(f"  Top-1 Accuracy : {overall_acc*100:.2f}%")
    print(f"  Top-5 Accuracy : {top5_acc*100:.2f}%")
    print(f"{'='*40}\n")

    report = classification_report(
        y_true_arr, y_pred_arr,
        target_names=EMOTIONS,
        digits=4,
    )
    print(report)

    # Save text report
    report_path = out_dir / f"report_{args.split}.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Top-1 Accuracy: {overall_acc*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc*100:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved  : {report_path}")

    # Save metrics JSON
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(range(NUM_CLASSES))
    )
    metrics = {
        "top1_accuracy": float(overall_acc),
        "top5_accuracy": float(top5_acc),
        "per_class": {
            emotion: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1[i]),
                "support":   int(support[i]),
            }
            for i, emotion in enumerate(EMOTIONS)
        },
    }
    json_path = out_dir / f"metrics_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON  : {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_confusion_matrix(y_true_arr, y_pred_arr,
                          out_dir / f"confusion_matrix_{args.split}.png")
    plot_per_class_accuracy(y_true_arr, y_pred_arr,
                            out_dir / f"per_class_accuracy_{args.split}.png")

    # ── Misclassified gallery ─────────────────────────────────────────────────
    print("\nSaving misclassified galleries...")
    save_misclassified(image_paths, y_true, y_pred, out_dir, args.top_k)

    print(f"\nAll outputs in: {out_dir.resolve()}")


NUM_CLASSES = len(EMOTIONS)

if __name__ == "__main__":
    main()
