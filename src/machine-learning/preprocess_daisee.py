"""
preprocess_daisee.py — Extract face crops from DAiSEE videos.

Run this ONCE as a Slurm CPU job (see slurm/preprocess.slurm) before training.
Output is written to CROPS_ROOT as:

    CROPS_ROOT/
        train/
            {ClipID}/
                frames.npy   shape: (N, C, H, W)  float32, values in [0,255]
                labels.npy   shape: (4,)           float32, values in [0,1]
        val/
            ...
        test/
            ...

Labels are rescaled from the DAiSEE 0-3 Likert scale to [0, 1] by dividing by 3.


We're look for faces in videos, and it must meet CFG.preproc.min_frames frames of faces for it to be used.
This outputs the frames (face focused and cropped)
"""

import argparse
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

from config import CFG, CROPS_ROOT, DAISEE_LABELS, DAISEE_VIDEOS, LABEL_COLS, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Worker — runs in a subprocess, one per video clip

def _init_worker(crop_size: int, margin: int, target_fps: int):
    """Initialise per-process state (MTCNN + config)."""
    global DETECTOR, CROP_SIZE, TARGET_FPS
    CROP_SIZE  = crop_size
    TARGET_FPS = target_fps
    DETECTOR = MTCNN( #MTCNN is for face retreival
        image_size=crop_size,
        margin=margin,
        keep_all=False,       # only the largest/most confident face
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,   # returns raw uint8 pixels, not normalised tensor
        device=torch.device("cpu"),
    )


def _process_clip(args) -> dict:
    """
    Extract face crops from one video clip.

    Returns a status dict so the parent process can aggregate statistics
    without needing shared memory.
    """
    video_path, label_vec, out_dir = args
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"path": str(video_path), "status": "error_open", "n_frames": 0}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / TARGET_FPS))

    crops = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            crop = DETECTOR(rgb)  # returns (C, H, W) uint8 tensor or None

            if crop is not None:
                # Store as (H, W, C) uint8 numpy — smaller on disk, convert at load time
                crops.append(crop.permute(1, 2, 0).numpy().astype(np.uint8))

        frame_idx += 1

    cap.release()

    if len(crops) < CFG.preproc.min_frames:
        return {
            "path": str(video_path),
            "status": "skipped_too_few",
            "n_frames": len(crops),
        }

    np.save(out_dir / "frames.npy", np.stack(crops))          # (N, H, W, C) uint8
    np.save(out_dir / "labels.npy", label_vec.astype(np.float32))  # (4,) float32

    return {"path": str(video_path), "status": "ok", "n_frames": len(crops)}


# Main

def preprocess_split(split: str, num_workers: int) -> None:
    label_csv  = DAISEE_LABELS[split]
    video_root = DAISEE_VIDEOS[split]
    out_root   = CROPS_ROOT / split

    log.info(f"Processing split: {split}")
    log.info(f"  Labels : {label_csv}")
    log.info(f"  Videos : {video_root}")
    log.info(f"  Output : {out_root}")

    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    df = pd.read_csv(label_csv)
    df.columns = df.columns.str.strip()

    # DAiSEE CSV column is named "ClipID" and looks like "0101010001.avi"
    # Videos live under {video_root}/{student_id}/{ClipID}
    # We search recursively since folder depth varies across dataset versions.
    log.info("  Indexing video files (this takes ~30 s for 9k clips)...")
    video_index = {p.name: p for p in video_root.rglob("*.avi")}
    log.info(f"  Found {len(video_index)} .avi files")

    tasks = []
    missing = 0

    for _, row in df.iterrows():
        clip_name = str(row["ClipID"])
        if not clip_name.endswith(".avi"):
            clip_name += ".avi"

        if clip_name not in video_index:
            missing += 1
            continue

        # Rescale 0-3 Likert labels to [0, 1]
        label_vec = np.array([row[col] for col in LABEL_COLS], dtype=np.float32) / 3.0
        clip_id   = Path(clip_name).stem
        out_dir   = out_root / clip_id

        # Skip already-processed clips (allows resuming interrupted jobs)
        if (out_dir / "frames.npy").exists() and (out_dir / "labels.npy").exists():
            continue

        tasks.append((video_index[clip_name], label_vec, out_dir))

    log.info(f"  {missing} clips missing from disk (will be skipped)")
    log.info(f"  {len(tasks)} clips to process")

    if not tasks:
        log.info("  Nothing to do — all clips already processed.")
        return

    # Parallel processing with a process pool
    init_args = (CFG.preproc.crop_size, CFG.preproc.margin, CFG.preproc.target_fps)

    ok = skipped = errors = 0
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_clip, tasks, chunksize=8),
            total=len(tasks),
            desc=f"  [{split}]",
        ):
            if result["status"] == "ok":
                ok += 1
            elif result["status"].startswith("skipped"):
                skipped += 1
                log.debug(f"  Skipped {result['path']} ({result['n_frames']} faces)")
            else:
                errors += 1
                log.warning(f"  Error on {result['path']}: {result['status']}")

    log.info(f"  Done — ok: {ok}, skipped: {skipped}, errors: {errors}")

    # Write a manifest for fast dataset loading
    manifest = []
    for clip_dir in sorted(out_root.glob("*")):
        frames_path = clip_dir / "frames.npy"
        labels_path = clip_dir / "labels.npy"
        if frames_path.exists() and labels_path.exists():
            n = np.load(frames_path, mmap_mode="r").shape[0]
            manifest.append({"clip_id": clip_dir.name, "n_frames": int(n)})

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"  Manifest written: {manifest_path} ({len(manifest)} clips)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess DAiSEE face crops")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which dataset splits to process")
    parser.add_argument("--workers", type=int, default=CFG.preproc.num_workers,
                        help="Number of parallel worker processes")
    args = parser.parse_args()


    CROPS_ROOT.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        preprocess_split(split, args.workers)

    log.info("All splits complete.")


if __name__ == "__main__":
    main()
