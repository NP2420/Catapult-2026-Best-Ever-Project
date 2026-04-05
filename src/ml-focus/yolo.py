"""
convert_to_yolo.py — Convert tired-data CSV annotations → YOLO format.

Run this LOCALLY before uploading to Gautschi.

Input structure (what Roboflow gave you):
    tired-data/
        train/
            _annotations.csv
            image1.jpg ...
        valid/
            _annotations.csv
            image1.jpg ...

Output structure (what YOLO needs):
    tired-yolo/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
        data.yaml

The valid/ split is divided 50/50 into val/ and test/ since no test split
was provided. The split is deterministic (seeded) so results are reproducible.

Usage:
    python convert_to_yolo.py
    python convert_to_yolo.py --src tired-data --dst tired-yolo
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Class ordering (alphabetical, matches Roboflow export) ──────────────────
CLASSES = ["closed_eye", "closed_mouth", "open_eye", "open_mouth"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# ── Val → (val, test) split ratio ───────────────────────────────────────────
TEST_FRACTION = 0.50   # 50% of valid becomes test, 50% stays as val
SEED = 42


def read_csv(csv_path: Path) -> dict[str, list[dict]]:
    """
    Returns {filename: [{"class", "xmin","ymin","xmax","ymax","w","h"}, ...]}
    """
    records = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["filename"]].append({
                "class": row["class"],
                "xmin":  int(row["xmin"]),
                "ymin":  int(row["ymin"]),
                "xmax":  int(row["xmax"]),
                "ymax":  int(row["ymax"]),
                "w":     int(row["width"]),
                "h":     int(row["height"]),
            })
    return dict(records)


def to_yolo_line(ann: dict) -> str:
    """Convert pixel bbox → normalised YOLO: class x_c y_c w h"""
    class_id = CLASS_TO_ID[ann["class"]]
    W, H = ann["w"], ann["h"]
    x_c = ((ann["xmin"] + ann["xmax"]) / 2) / W
    y_c = ((ann["ymin"] + ann["ymax"]) / 2) / H
    bw  = (ann["xmax"] - ann["xmin"]) / W
    bh  = (ann["ymax"] - ann["ymin"]) / H
    return f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"


def write_split(
    filenames: list[str],
    annotations: dict,
    src_image_dir: Path,
    dst_dir: Path,
) -> int:
    """Copy images + write label txts for one split. Returns count written."""
    img_dir = dst_dir / "images"
    lbl_dir = dst_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fname in filenames:
        src_img = src_image_dir / fname
        if not src_img.exists():
            print(f"  [WARN] image not found, skipping: {src_img}")
            continue

        # Copy image
        shutil.copy2(src_img, img_dir / fname)

        # Write label file (empty txt if no annotations — valid for YOLO)
        label_path = lbl_dir / (Path(fname).stem + ".txt")
        lines = [to_yolo_line(a) for a in annotations.get(fname, [])]
        label_path.write_text("\n".join(lines))
        written += 1

    return written


def write_yaml(dst_root: Path, yaml_path: Path) -> None:
    abs_root = dst_root.resolve()
    yaml_path.write_text(
        f"path: {abs_root}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    print(f"  data.yaml → {yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="tired-data",  help="Path to Roboflow download")
    parser.add_argument("--dst", default="tired-yolo",  help="Output YOLO dataset root")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}  —  run from the directory containing tired-data/")

    print(f"Source : {src.resolve()}")
    print(f"Output : {dst.resolve()}\n")

    # ── Train split ──────────────────────────────────────────────────────────
    train_csv  = src / "train" / "_annotations.csv"
    train_anns = read_csv(train_csv)
    train_imgs = sorted(train_anns.keys())

    n_train = write_split(train_imgs, train_anns, src / "train", dst / "train")
    print(f"Train : {n_train} images written")

    # ── Valid → val + test ───────────────────────────────────────────────────
    valid_csv  = src / "valid" / "_annotations.csv"
    valid_anns = read_csv(valid_csv)
    valid_imgs = sorted(valid_anns.keys())

    random.seed(SEED)
    random.shuffle(valid_imgs)
    split_idx  = int(len(valid_imgs) * TEST_FRACTION)
    test_imgs  = valid_imgs[:split_idx]
    val_imgs   = valid_imgs[split_idx:]

    n_val  = write_split(val_imgs,  valid_anns, src / "valid", dst / "val")
    n_test = write_split(test_imgs, valid_anns, src / "valid", dst / "test")
    print(f"Val   : {n_val}  images written")
    print(f"Test  : {n_test} images written")

    # ── Class distribution report ────────────────────────────────────────────
    print("\nClass distribution (train):")
    counts = defaultdict(int)
    for anns in train_anns.values():
        for a in anns:
            counts[a["class"]] += 1
    for cls in CLASSES:
        print(f"  {cls:<15} {counts[cls]:>5} boxes")

    # ── data.yaml ────────────────────────────────────────────────────────────
    write_yaml(dst, dst / "data.yaml")

    print(f"\nDone. Upload tired-yolo/ to Gautschi:")
    print(f"  rsync -av tired-yolo/ gautschi.rcac.purdue.edu:"
          f"/scratch/gautschi/pham191/tired-model/tired-yolo/")


if __name__ == "__main__":
    main()