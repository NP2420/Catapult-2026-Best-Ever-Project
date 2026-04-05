"""
webcam_test.py
--------------
Real-time emotion detection from your webcam.

⚠️  Run this LOCALLY (your laptop/desktop), not on the cluster.
    After training, copy best.pt to your machine:
        scp pham191@gautschi.rcac.purdue.edu:\
            /scratch/gautschi/pham191/Catapult-2026-Best-Ever-Project/\
            src/ml-emotions/outputs/emotion_yolov8x/weights/best.pt \
            ./best.pt

Requirements (local):
    pip install ultralytics opencv-python pillow numpy

Usage:
    python webcam_test.py --model best.pt
    python webcam_test.py --model best.pt --camera 1  # if built-in cam is index 1
    python webcam_test.py --model best.pt --video path/to/file.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ─── Config ──────────────────────────────────────────────────────────────────
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS_BGR = {
    "anger":    (0,   0,   220),   # Red
    "contempt": (153, 51,  153),   # Purple
    "disgust":  (0,   153, 51),    # Green
    "fear":     (0,   153, 220),   # Orange
    "happy":    (0,   215, 255),   # Yellow
    "neutral":  (150, 150, 150),   # Gray
    "sad":      (180, 80,  20),    # Blue
    "surprise": (180, 200, 20),    # Teal
}

EMOTION_EMOJI = {
    "anger": "😠", "contempt": "😤", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲",
}

# Face detector — OpenCV DNN (more accurate than Haar)
FACE_MODEL_URL_CFG  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL_WEIGHTS = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
FACE_CFG     = "face_detector.prototxt"
FACE_WEIGHTS = "face_detector.caffemodel"


def download_face_model():
    """Download OpenCV DNN face detector if not present."""
    import urllib.request
    for url, fname in [(FACE_MODEL_URL_CFG, FACE_CFG), (FACE_MODEL_URL_WEIGHTS, FACE_WEIGHTS)]:
        if not Path(fname).exists():
            print(f"Downloading {fname}...")
            try:
                urllib.request.urlretrieve(url, fname)
            except Exception as e:
                print(f"  [WARN] Failed to download {fname}: {e}")
                return False
    return True


def load_face_detector():
    """Load face detector — prefers DNN, falls back to Haar cascade."""
    if download_face_model() and Path(FACE_CFG).exists() and Path(FACE_WEIGHTS).exists():
        net = cv2.dnn.readNetFromCaffe(FACE_CFG, FACE_WEIGHTS)
        print("Face detector: OpenCV DNN (SSD ResNet-10)")
        return ("dnn", net)
    else:
        # Fallback to Haar cascade (built into OpenCV)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("Face detector: Haar Cascade (fallback)")
        return ("haar", cascade)


def detect_faces_dnn(net, frame, conf_threshold=0.5):
    """Detect faces using the DNN model. Returns list of (x, y, w, h) boxes."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        faces.append((x1, y1, x2 - x1, y2 - y1, conf))
    return faces


def detect_faces_haar(cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                    minSize=(60, 60))
    return [(x, y, w, h, 1.0) for (x, y, w, h) in dets] if len(dets) else []


def draw_emotion_bar(frame, x, y, w, emotion_probs):
    """Draw a small probability bar chart on the frame."""
    bar_x = x + w + 10
    bar_y = y
    bar_w = 150
    bar_h = 16
    pad = 2

    if bar_x + bar_w > frame.shape[1]:
        bar_x = x - bar_w - 10
    if bar_x < 0:
        return

    for i, (emo, prob) in enumerate(emotion_probs):
        color = EMOTION_COLORS_BGR[emo]
        ey = bar_y + i * (bar_h + pad)
        if ey + bar_h > frame.shape[0]:
            break
        filled = int(prob * bar_w)
        cv2.rectangle(frame, (bar_x, ey), (bar_x + bar_w, ey + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, ey), (bar_x + filled, ey + bar_h), color, -1)
        label = f"{emo[:7]:<7} {prob*100:4.1f}%"
        cv2.putText(frame, label, (bar_x + 3, ey + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def run_webcam(model: YOLO, detector, camera_idx: int, video_src, conf_threshold: float,
               imgsz: int, device: str):
    src = video_src if video_src else camera_idx
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {src}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector_type, detector_obj = detector

    fps_history = []
    frame_idx = 0
    print("\nPress 'q' to quit | 's' to save screenshot | 'f' to toggle face conf")
    show_conf = True

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        frame_idx += 1

        # ── Detect faces ───────────────────────────────────────────────────
        if detector_type == "dnn":
            faces = detect_faces_dnn(detector_obj, frame, conf_threshold)
        else:
            faces = detect_faces_haar(detector_obj, frame)

        # ── Classify each face ────────────────────────────────────────────
        for (x, y, w, h, face_conf) in faces:
            # Pad crop slightly
            pad_px = int(0.1 * max(w, h))
            x1 = max(0, x - pad_px)
            y1 = max(0, y - pad_px)
            x2 = min(frame.shape[1], x + w + pad_px)
            y2 = min(frame.shape[0], y + h + pad_px)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Convert to PIL for YOLO
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

            result = model.predict(
                source=face_pil,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )[0]

            probs = result.probs.data.cpu().numpy()
            top_idx = int(np.argmax(probs))
            top_emotion = EMOTIONS[top_idx]
            top_conf = float(probs[top_idx])

            color = EMOTION_COLORS_BGR[top_emotion]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Emotion label above box
            label = f"{EMOTION_EMOJI.get(top_emotion, '')} {top_emotion.upper()}  {top_conf*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(frame, (x, y - lh - 10), (x + lw + 8, y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            # Face detection confidence
            if show_conf:
                cv2.putText(frame, f"face: {face_conf:.2f}", (x, y + h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Probability bar chart
            emotion_probs_sorted = sorted(
                [(EMOTIONS[i], float(probs[i])) for i in range(len(EMOTIONS))],
                key=lambda t: t[1], reverse=True
            )
            draw_emotion_bar(frame, x, y, w, emotion_probs_sorted)

        # ── FPS overlay ───────────────────────────────────────────────────
        t_end = time.time()
        fps = 1.0 / (t_end - t_start + 1e-9)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Emotion Detection  [q=quit  s=screenshot  f=toggle-conf]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_name = f"screenshot_{frame_idx:06d}.jpg"
            cv2.imwrite(out_name, frame)
            print(f"Screenshot saved: {out_name}")
        elif key == ord("f"):
            show_conf = not show_conf

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


def parse_args():
    p = argparse.ArgumentParser(description="Webcam emotion detection")
    p.add_argument("--model",     type=str,   required=True,
                   help="Path to best.pt (copy from cluster)")
    p.add_argument("--camera",    type=int,   default=0,
                   help="Webcam index (0 = default)")
    p.add_argument("--video",     type=str,   default=None,
                   help="Path to video file instead of webcam")
    p.add_argument("--imgsz",     type=int,   default=224)
    p.add_argument("--device",    type=str,   default="cpu",
                   help="'cpu' for local laptop, '0' if you have a local GPU")
    p.add_argument("--face_conf", type=float, default=0.5,
                   help="Minimum face detection confidence")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    assert model_path.exists(), (
        f"Model not found: {model_path}\n"
        f"Copy it from the cluster:\n"
        f"  scp pham191@gautschi.rcac.purdue.edu:"
        f"/scratch/gautschi/pham191/Catapult-2026-Best-Ever-Project/"
        f"src/ml-emotions/best.pt ."
    )

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print("Loading face detector...")
    detector = load_face_detector()

    run_webcam(
        model=model,
        detector=detector,
        camera_idx=args.camera,
        video_src=args.video,
        conf_threshold=args.face_conf,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
