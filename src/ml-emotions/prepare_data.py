"""
prepare_data.py
---------------
Fixes the AffectNet val/ directory (mixed case like 'Anger'/'anger')
by merging duplicates into a single lowercase folder per class.
Run ONCE before training.

Usage:
    python prepare_data.py --data_dir ./data
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import json


EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def merge_case_duplicates(split_dir: Path, dry_run: bool = False):
    """
    Scan a split directory (train/ or val/) and merge any case-duplicate
    folders (e.g., 'Anger' → 'anger').
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {split_dir}")
    folder_map = defaultdict(list)

    for item in split_dir.iterdir():
        if item.is_dir():
            folder_map[item.name.lower()].append(item)

    for canonical_name, folders in folder_map.items():
        if canonical_name not in EMOTIONS:
            print(f"  [WARN] Unknown emotion folder: {[f.name for f in folders]} — skipping")
            continue

        canonical_path = split_dir / canonical_name

        # Separate: is the correctly-cased folder already there?
        exact_match = [f for f in folders if f.name == canonical_name]
        others      = [f for f in folders if f.name != canonical_name]

        if not exact_match:
            # Rename the first occurrence to lowercase
            source = others.pop(0)
            print(f"  Renaming {source.name!r} → {canonical_name!r}")
            if not dry_run:
                source.rename(canonical_path)

        # Merge any remaining mismatched-case folders into canonical
        for extra in others:
            print(f"  Merging  {extra.name!r} → {canonical_name!r}")
            if not dry_run:
                for img in extra.iterdir():
                    dest = canonical_path / img.name
                    # Avoid collisions by appending a suffix
                    if dest.exists():
                        stem, suffix = img.stem, img.suffix
                        dest = canonical_path / f"{stem}_dup{suffix}"
                    shutil.move(str(img), dest)
                extra.rmdir()

    # Verify all expected folders exist
    missing = [e for e in EMOTIONS if not (split_dir / e).exists()]
    if missing:
        print(f"  [ERROR] Missing emotion folders: {missing}")
    else:
        print(f"  ✓ All {len(EMOTIONS)} emotion folders present.")


def count_images(data_dir: Path):
    """Print image count per split/class and return summary dict."""
    summary = {}
    print("\n=== Dataset Summary ===")
    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  [WARN] {split}/ not found — skipping")
            continue
        split_total = 0
        print(f"\n  [{split}]")
        for emotion in EMOTIONS:
            emo_dir = split_dir / emotion
            if emo_dir.exists():
                count = len(list(emo_dir.glob("*.*")))
                print(f"    {emotion:<12}: {count:>6} images")
                summary[f"{split}/{emotion}"] = count
                split_total += count
            else:
                print(f"    {emotion:<12}: MISSING")
                summary[f"{split}/{emotion}"] = 0
        print(f"    {'TOTAL':<12}: {split_total:>6} images")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare AffectNet dataset for YOLOv8")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data/ directory containing train/ and val/")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview changes without modifying files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    print("=" * 60)
    print("AffectNet Data Preparation")
    print("=" * 60)

    for split in ["train", "val"]:
        split_dir = data_dir / split
        if split_dir.exists():
            merge_case_duplicates(split_dir, dry_run=args.dry_run)
        else:
            print(f"\n[WARN] {split}/ directory not found — skipping")

    summary = count_images(data_dir)

    if not args.dry_run:
        summary_path = data_dir / "dataset_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    print("\nDone! Ready to train.")


if __name__ == "__main__":
    main()
