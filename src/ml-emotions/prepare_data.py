"""
prepare_data.py
---------------
Reorganizes the AffectNet dataset using labels.csv as ground truth.

The raw data has images filed under emotion folders, but labels.csv
contains the ACTUAL label (which often differs from the folder name)
plus a face-confidence score (relFCs) that can be used to filter out
low-quality / ambiguous samples.

What this script does:
  1. Read labels.csv
  2. Filter rows below --min_conf threshold
  3. Copy/symlink images into a clean directory structure:
       data_clean/
           train/  anger/ contempt/ ... (per CSV label)
           val/    anger/ contempt/ ...
  4. Print a summary with label distribution and mismatch stats

Usage:
    # Preview — no files touched
    python prepare_data.py --dry_run

    # Build data_clean/ with default confidence threshold (0.7)
    python prepare_data.py

    # Stricter filter
    python prepare_data.py --min_conf 0.80

    # Lower filter (more data, noisier)
    python prepare_data.py --min_conf 0.50

    # Use symlinks instead of copies (saves disk space)
    python prepare_data.py --symlink
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare AffectNet using labels.csv")
    p.add_argument("--data_dir",   type=str, default="./data",
                   help="Directory containing labels.csv, train/, val/")
    p.add_argument("--out_dir",    type=str, default="./data_clean",
                   help="Output directory for reorganized dataset")
    p.add_argument("--min_conf",   type=float, default=0.70,
                   help="Minimum relFCs score to keep an image (0.0 = keep all)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Show stats without copying any files")
    p.add_argument("--symlink",    action="store_true",
                   help="Create symlinks instead of copies (saves disk space)")
    p.add_argument("--val_split",  type=float, default=None,
                   help="If CSV has no split info, fraction to use as val (e.g. 0.15). "
                        "Leave None to infer split from the source subfolder name.")
    return p.parse_args()


def load_csv(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "labels.csv"
    assert csv_path.exists(), f"labels.csv not found in {data_dir}"
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded labels.csv: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df


def resolve_image_path(data_dir: Path, pth: str):
    """
    Try train/ then val/ to find the actual file.
    Returns (resolved_path, split_name) or (None, None).
    """
    for split in ("train", "val"):
        candidate = data_dir / split / pth
        if candidate.exists():
            return candidate, split
    return None, None


def build_dataset(df: pd.DataFrame, data_dir: Path, out_dir: Path,
                  min_conf: float, dry_run: bool, symlink: bool, val_split):
    """
    Core logic: iterate CSV rows, filter, resolve paths, copy/link.
    """
    stats = {
        "total":          len(df),
        "below_conf":     0,
        "not_found":      0,
        "label_mismatch": 0,
        "copied":         0,
        "by_split_label": defaultdict(Counter),
    }

    # Pre-create output directories
    if not dry_run:
        for split in ("train", "val"):
            for emotion in EMOTIONS:
                (out_dir / split / emotion).mkdir(parents=True, exist_ok=True)

    for row in df.itertuples():
        pth: str    = row.pth
        label: str  = row.label
        conf: float = row.relFCs

        # ── Filter by confidence ──────────────────────────────────────────
        if conf < min_conf:
            stats["below_conf"] += 1
            continue

        # ── Validate label ────────────────────────────────────────────────
        label = label.strip().lower()
        if label not in EMOTIONS:
            print(f"  [WARN] Unknown label '{label}' for {pth} — skipping")
            continue

        # ── Resolve file path ─────────────────────────────────────────────
        src, split = resolve_image_path(data_dir, pth)
        if src is None:
            stats["not_found"] += 1
            continue

        # Track folder vs CSV label mismatches
        folder_label = Path(pth).parts[0].lower()
        if folder_label != label:
            stats["label_mismatch"] += 1

        # ── Determine split ───────────────────────────────────────────────
        if val_split is not None:
            import random
            split = "val" if random.random() < val_split else "train"

        stats["by_split_label"][split][label] += 1
        stats["copied"] += 1

        if dry_run:
            continue

        # ── Copy or symlink ───────────────────────────────────────────────
        dest = out_dir / split / label / src.name
        # Handle filename collisions (same name across source folders)
        if dest.exists():
            stem, suffix = src.stem, src.suffix
            counter = 1
            while dest.exists():
                dest = out_dir / split / label / f"{stem}_{counter}{suffix}"
                counter += 1

        if symlink:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)

    return stats


def print_summary(stats: dict, min_conf: float):
    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"  Total rows in CSV      : {stats['total']:>8}")
    print(f"  Filtered (conf < {min_conf:.2f}) : {stats['below_conf']:>8}")
    print(f"  File not found         : {stats['not_found']:>8}")
    print(f"  Label↔folder mismatch  : {stats['label_mismatch']:>8}  "
          f"({'corrected by CSV' if stats['copied'] > 0 else 'dry-run'})")
    print(f"  Images kept            : {stats['copied']:>8}")

    for split in ("train", "val"):
        counts = stats["by_split_label"].get(split, {})
        if not counts:
            continue
        total = sum(counts.values())
        print(f"\n  [{split}]  total={total}")
        for emotion in EMOTIONS:
            n = counts.get(emotion, 0)
            bar = "█" * (n // max(1, total // 40))
            print(f"    {emotion:<12}: {n:>6}  {bar}")

    # Class imbalance warning
    for split in ("train", "val"):
        counts = stats["by_split_label"].get(split, {})
        if len(counts) < 2:
            continue
        mn, mx = min(counts.values()), max(counts.values())
        ratio = mx / mn if mn > 0 else float("inf")
        if ratio > 5:
            print(f"\n  ⚠  [{split}] imbalance ratio {ratio:.1f}x — "
                  f"consider weighted loss in training")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print("=" * 60)
    print(f"AffectNet Data Preparation (labels.csv mode)")
    print(f"  Source    : {data_dir}")
    print(f"  Output    : {out_dir}  {'[DRY RUN]' if args.dry_run else ''}")
    print(f"  Min conf  : {args.min_conf}")
    print(f"  Mode      : {'symlink' if args.symlink else 'copy'}")
    print("=" * 60)

    df = load_csv(data_dir)

    # Show label-vs-folder mismatch rate up front
    folder_labels = df["pth"].apply(lambda p: Path(p).parts[0].lower())
    csv_labels    = df["label"].str.strip().str.lower()
    mismatch_pct  = (folder_labels != csv_labels).mean() * 100
    print(f"\nLabel-vs-folder mismatch rate: {mismatch_pct:.1f}%  "
          f"← this is why we use the CSV, not folder names")

    # Confidence score distribution
    print(f"\nrelFCs stats:")
    print(f"  min={df['relFCs'].min():.3f}  "
          f"max={df['relFCs'].max():.3f}  "
          f"mean={df['relFCs'].mean():.3f}  "
          f"median={df['relFCs'].median():.3f}")
    kept_pct = (df["relFCs"] >= args.min_conf).mean() * 100
    print(f"  Images kept at conf≥{args.min_conf}: {kept_pct:.1f}%")

    stats = build_dataset(
        df        = df,
        data_dir  = data_dir,
        out_dir   = out_dir,
        min_conf  = args.min_conf,
        dry_run   = args.dry_run,
        symlink   = args.symlink,
        val_split = args.val_split,
    )

    print_summary(stats, args.min_conf)

    if not args.dry_run:
        config = {
            "data_dir":  str(data_dir.resolve()),
            "out_dir":   str(out_dir.resolve()),
            "min_conf":  args.min_conf,
            "symlink":   args.symlink,
            "emotions":  EMOTIONS,
            "stats": {
                "total":          stats["total"],
                "kept":           stats["copied"],
                "below_conf":     stats["below_conf"],
                "label_mismatch": stats["label_mismatch"],
            },
        }
        cfg_path = out_dir / "prep_config.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved: {cfg_path}")
        print(f"\n✓ Done. Point train.py at: {out_dir}")
        print(f"  python train.py --data_dir {out_dir}")
    else:
        print("\n[DRY RUN complete — no files modified]")
        print(f"Re-run without --dry_run to build {out_dir}")


if __name__ == "__main__":
    main()