"""
realtime_inference.py — Webcam tiredness detection using trained YOLOv8 model.

Runs on the DEMO LAPTOP. Requires only:
    best.pt (copy from Gautschi after training)
    ultralytics, opencv-python

Usage:
    python realtime_inference.py
    python realtime_inference.py --ckpt path/to/best.pt --cam 0
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["closed_eye", "closed_mouth", "open_eye", "open_mouth"]

# BGR colors for bounding boxes
BOX_COLORS = {
    "closed_eye":    (74,  75, 226),   # red-ish
    "closed_mouth":  (23, 117, 186),   # amber-ish
    "open_eye":      (117, 158, 29),   # teal
    "open_mouth":    (183,  74,  83),  # purple
}

EYE_WEIGHT   = 0.7
MOUTH_WEIGHT = 0.3
EMA_ALPHA    = 0.15    # smoothing — lower = smoother but more lag
WINDOW_SECS  = 5.0     # rolling window for time-averaged score (shown in HUD)


def tiredness_from_boxes(class_names: list[str]) -> float:
    closed_eyes   = class_names.count("closed_eye")
    open_eyes     = class_names.count("open_eye")
    open_mouths   = class_names.count("open_mouth")
    closed_mouths = class_names.count("closed_mouth")

    eye_score   = closed_eyes  / max(closed_eyes  + open_eyes,    1)
    mouth_score = open_mouths  / max(open_mouths  + closed_mouths, 1)
    return EYE_WEIGHT * eye_score + MOUTH_WEIGHT * mouth_score


def draw_hud(frame: np.ndarray, score: float, rolling_score: float,
             fps: float, face_found: bool) -> np.ndarray:
    h, w = frame.shape[:2]

    # Determine state
    if not face_found:
        state, bar_color = "No face", (120, 120, 120)
    elif score < 0.35:
        state, bar_color = "Awake", (100, 200, 100)
    elif score < 0.65:
        state, bar_color = "Drowsy", (50, 165, 220)
    else:
        state, bar_color = "TIRED", (60, 60, 230)

    # Semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (300, 145), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Score bar background
    BAR_X, BAR_Y, BAR_W, BAR_H = 16, 48, 252, 22
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H), (60, 60, 60), -1)

    # Score bar fill
    fill = int(score * BAR_W)
    if fill > 0:
        cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + fill, BAR_Y + BAR_H), bar_color, -1)

    # Bar border
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H), (180, 180, 180), 1)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Tiredness: {score:.2f}", (16, 40),
                font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{score:.0%}", (BAR_X + fill - 30, BAR_Y + 16),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"State  : {state}", (16, 90),
                font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"5s avg : {rolling_score:.2f}", (16, 112),
                font, 0.5,  (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS    : {fps:.0f}", (16, 132),
                font, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    # Big state badge in top-right
    badge_text = state.upper()
    (tw, th), _ = cv2.getTextSize(badge_text, font, 0.9, 2)
    bx = w - tw - 14
    cv2.rectangle(frame, (bx - 8, 10), (w - 6, th + 22), bar_color, -1)
    cv2.putText(frame, badge_text, (bx, th + 14),
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best.pt",
                        help="Path to trained best.pt")
    parser.add_argument("--cam",  type=int, default=0,
                        help="Webcam device index")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Detection confidence threshold")
    parser.add_argument("--iou",  type=float, default=0.45)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Model not found: {ckpt}\n"
            "Copy best.pt from Gautschi:\n"
            "  scp gautschi.rcac.purdue.edu:/scratch/gautschi/pham191/tired-model/"
            "runs/detect/tired_yolov8m/weights/best.pt ."
        )

    print(f"Loading model: {ckpt}")
    model = YOLO(str(ckpt))
    model.fuse()   # fuse Conv+BN for faster CPU inference

    print(f"Opening camera {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam}")

    ema_score     = 0.5          # current EMA tiredness score
    score_history: deque[tuple[float, float]] = deque()   # (timestamp, score)
    prev_time     = time.time()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            stream=False,
        )[0]

        face_found  = results.boxes is not None and len(results.boxes) > 0
        class_names = [CLASSES[int(c)] for c in results.boxes.cls] if face_found else []

        # ── Tiredness score + EMA ─────────────────────────────────────────────
        raw_score = tiredness_from_boxes(class_names)
        ema_score = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * ema_score

        # Rolling average over last WINDOW_SECS seconds
        now = time.time()
        score_history.append((now, ema_score))
        while score_history and score_history[0][0] < now - WINDOW_SECS:
            score_history.popleft()
        rolling_score = (sum(s for _, s in score_history) / len(score_history)
                         if score_history else ema_score)

        # ── Draw detection boxes ──────────────────────────────────────────────
        if face_found:
            for box in results.boxes:
                cls_name = CLASSES[int(box.cls)]
                conf_val = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = BOX_COLORS.get(cls_name, (128, 128, 128))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name.replace('_',' ')} {conf_val:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps = 1.0 / max(now - prev_time, 1e-4)
        prev_time = now

        # ── HUD overlay ───────────────────────────────────────────────────────
        frame = draw_hud(frame, ema_score, rolling_score, fps, face_found)

        cv2.imshow("Tiredness Monitor  [q to quit]", frame)

        # Console log every ~3 seconds
        if int(now) % 3 == 0:
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"score={ema_score:.2f}  "
                  f"5s_avg={rolling_score:.2f}  "
                  f"detected={class_names}    ", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()