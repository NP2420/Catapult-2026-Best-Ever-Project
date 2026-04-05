"""
realtime_inference.py — Real-time engagement detection from a webcam feed.

Runs on the DEMO LAPTOP (not on Gautschi). Requires only:
  - clippy_engagement_int8.onnx (copy from Gautschi after training)
  - onnxruntime, opencv-python, facenet-pytorch

Architecture of this module:
  - A background thread continuously reads webcam frames and buffers face crops
  - The inference thread processes the buffer every EMIT_INTERVAL seconds
  - Results are pushed onto a thread-safe queue consumed by the main app

Usage as a standalone test:
    python realtime_inference.py

Usage from main.py:
    from realtime_inference import InferenceEngine
    engine = InferenceEngine(output_queue)
    engine.start()
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from facenet_pytorch import MTCNN

from config import CFG, LABEL_NAMES


# EngagementState dataclass — the unit of output from this module

@dataclass
class EngagementState:
    timestamp:    float
    boredom:      float
    engagement:   float
    confusion:    float
    frustration:  float
    confidence:   float   # MTCNN detection probability of the most recent face
    face_detected: bool

    def to_dict(self) -> dict:
        return {
            "timestamp":    self.timestamp,
            "boredom":      round(self.boredom, 3),
            "engagement":   round(self.engagement, 3),
            "confusion":    round(self.confusion, 3),
            "frustration":  round(self.frustration, 3),
            "confidence":   round(self.confidence, 3),
            "face_detected": self.face_detected,
        }

    def dominant(self) -> str:
        """Return the name of the strongest signal."""
        scores = {
            "engaged":    self.engagement,
            "bored":      self.boredom,
            "confused":   self.confusion,
            "frustrated": self.frustration,
        }
        return max(scores, key=scores.get)


# Image preprocessing helpers

_MEAN = np.array(CFG.inference.mean, dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array(CFG.inference.std,  dtype=np.float32).reshape(3, 1, 1)


def _normalise(crop_hwc_uint8: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, C) uint8 → (C, H, W) float32, ImageNet normalised.
    """
    chw = crop_hwc_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0
    return (chw - _MEAN) / _STD


# HUD drawing

_HUD_COLORS = {
    "boredom":    (200, 200,  50),   # BGR
    "engagement": ( 50, 200,  50),
    "confusion":  (200, 100,  50),
    "frustration":(  50,  50, 200),
}

def draw_hud(frame: np.ndarray, state: EngagementState) -> np.ndarray:
    """
    Draw a semi-transparent emotion bar chart in the top-left corner,
    plus a binary 'Focused vs Unfocused' panel.
    Works in-place on the frame; returns the frame for chaining.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background panel for everything
    panel_w, panel_h = 220, 140
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # ---------------------------
    # Binary Focus/Unfocused panel
    # ---------------------------
    bar_x  = 20
    bar_y  = 15
    bar_h  = 18
    max_w  = 180

    focused = state.engagement
    unfocused = state.boredom + state.confusion + state.frustration
    total = focused + unfocused
    # avoid divide-by-zero
    if total > 0:
        focused /= total
        unfocused /= total
    else:
        focused, unfocused = 0.0, 0.0

    # Focus bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_w, bar_y + bar_h),
                  (70, 70, 70), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(focused * max_w), bar_y + bar_h),
                  (50, 200, 50), -1)
    cv2.putText(frame, f"Focused  {focused:.2f}",
                (bar_x + 4, bar_y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    # Unfocused bar
    bar_y += bar_h + 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_w, bar_y + bar_h),
                  (70, 70, 70), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(unfocused * max_w), bar_y + bar_h),
                  (200, 50, 50), -1)
    cv2.putText(frame, f"Unfocused {unfocused:.2f}",
                (bar_x + 4, bar_y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    # ---------------------------
    # Regular emotion bars
    # ---------------------------
    bar_y0 = bar_y + bar_h + 12
    gap    = 22
    for i, name in enumerate(LABEL_NAMES):
        score = getattr(state, name)
        y     = bar_y0 + i * gap
        color = _HUD_COLORS.get(name, (180, 180, 180))

        # Background track
        cv2.rectangle(frame, (bar_x, y), (bar_x + max_w, y + bar_h),
                      (70, 70, 70), -1)
        # Filled portion
        fill_w = int(score * max_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill_w, y + bar_h),
                          color, -1)
        # Label
        cv2.putText(frame, f"{name[:4]}  {score:.2f}",
                    (bar_x + 4, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                    cv2.LINE_AA)

    # Dominant state
    cv2.putText(frame, f"State: {state.dominant()}",
                (bar_x, bar_y0 + 4 * gap + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    return frame


# Core inference engine

class InferenceEngine:
    """
    Manages a webcam capture thread and an inference thread.

    Parameters
    ----------
    output_queue : queue.Queue
        Receives EngagementState objects every ~emit_interval seconds.
    cam_index : int
        OpenCV camera device index (0 = first webcam).
    show_hud : bool
        If True, opens a cv2 window with live feed + emotion bars.
    """

    def __init__(
        self,
        output_queue: queue.Queue,
        cam_index:    int  = CFG.inference.cam_index,
        show_hud:     bool = CFG.inference.display_hud,
    ):
        self.output_queue = output_queue
        self.cam_index    = cam_index
        self.show_hud     = show_hud

        # Load ONNX model
        onnx_path = CFG.inference.onnx_path
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run export_onnx.py on Gautschi, then copy the file here."
            )
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        print(f"[InferenceEngine] ONNX model loaded: {onnx_path}")

        # Face detector — CPU, no GPU required on demo laptop
        self.detector = MTCNN(
            image_size=CFG.preproc.crop_size,
            margin=CFG.preproc.margin,
            keep_all=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,   # raw uint8 pixels
            device=torch.device("cpu"),
        )

        # Rolling frame buffer: stores normalised (C, H, W) crops
        self.frame_buffer: deque = deque(maxlen=CFG.model.seq_len)
        self.ema_state:    Optional[np.ndarray] = None

        self._stop_event   = threading.Event()
        self._latest_frame = None
        self._frame_lock   = threading.Lock()

        # Track face detection confidence for emitted state
        self._last_confidence = 0.0


    def start(self):
        """Start capture and inference threads. Non-blocking."""
        self._capture_thread   = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._capture_thread.start()
        self._inference_thread.start()
        print("[InferenceEngine] Started")

    def stop(self):
        self._stop_event.set()
        self._capture_thread.join(timeout=3)
        self._inference_thread.join(timeout=3)
        cv2.destroyAllWindows()


    def _capture_loop(self):
        """
        Thread 1: reads webcam frames, detects faces, resizes to model input,
        and appends normalized crops to the rolling buffer.
        """
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {self.cam_index}")
            self._stop_event.set()
            return

        # Request 30fps from the webcam driver
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        EXTRACT_INTERVAL = max(1, round(30 / CFG.preproc.target_fps))

        while not self._stop_event.is_set():
            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = bgr.copy()

            if frame_count % EXTRACT_INTERVAL == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                crop, prob = self.detector(rgb, return_prob=True)

                if crop is not None and (prob is None or prob > CFG.inference.min_confidence):
                    # Convert tensor → numpy
                    crop_hwc = crop.permute(1, 2, 0).numpy().astype(np.uint8)

                    # Resize to ONNX model input
                    crop_hwc = cv2.resize(crop_hwc, 
                                        (CFG.preproc.crop_size, CFG.preproc.crop_size))

                    # Normalize and append
                    self.frame_buffer.append(_normalise(crop_hwc))
                    self._last_confidence = float(prob) if prob is not None else 1.0

            frame_count += 1

        cap.release()


    def _inference_loop(self):
        """
        Thread 2: every emit_interval seconds, run the ONNX model over the
        current frame buffer and push an EngagementState to the output queue.
        """
        last_emit = time.time()

        while not self._stop_event.is_set():
            now = time.time()

            # Draw HUD on main thread display
            if self.show_hud:
                with self._frame_lock:
                    frame = self._latest_frame.copy() if self._latest_frame is not None else None
                if frame is not None:
                    if self.ema_state is not None:
                        dummy_state = EngagementState(
                            timestamp=now,
                            boredom=float(self.ema_state[0]),
                            engagement=float(self.ema_state[1]),
                            confusion=float(self.ema_state[2]),
                            frustration=float(self.ema_state[3]),
                            confidence=self._last_confidence,
                            face_detected=len(self.frame_buffer) > 0,
                        )
                        draw_hud(frame, dummy_state)
                    cv2.imshow("Clippy — Engagement Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop_event.set()
                        break

            if (now - last_emit) < CFG.inference.emit_interval:
                time.sleep(0.05)
                continue

            last_emit = now

            # Not enough frames yet
            if len(self.frame_buffer) < CFG.model.seq_len:
                self.output_queue.put(EngagementState(
                    timestamp=now,
                    boredom=0.0, engagement=0.5, confusion=0.0, frustration=0.0,
                    confidence=0.0, face_detected=False,
                ))
                continue

            # Stack buffer → (1, seq_len, C, H, W)
            frames_np = np.stack(list(self.frame_buffer))[None].astype(np.float32)

            # ONNX inference
            raw_scores = self.sess.run(None, {"frames": frames_np})[0][0]  # (4,)

            # EMA smoothing — prevents flickering between states
            alpha = CFG.inference.ema_alpha
            if self.ema_state is None:
                self.ema_state = raw_scores.copy()
            else:
                self.ema_state = alpha * raw_scores + (1 - alpha) * self.ema_state

            state = EngagementState(
                timestamp=now,
                boredom=float(self.ema_state[0]),
                engagement=float(self.ema_state[1]),
                confusion=float(self.ema_state[2]),
                frustration=float(self.ema_state[3]),
                confidence=self._last_confidence,
                face_detected=True,
            )
            self.output_queue.put(state)


# Standalone test — run this file directly to verify everything works

def inference_loop(output_queue: queue.Queue):
    """Entry point for main.py to call in a thread."""
    engine = InferenceEngine(output_queue)
    engine.start()
    # Block until stop event (engine runs in background threads)
    try:
        while not engine._stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    print("Starting real-time inference test. Press 'q' in the video window to quit.")
    q = queue.Queue()
    engine = InferenceEngine(q, show_hud=True)
    engine.start()

    try:
        while True:
            try:
                state = q.get(timeout=0.5)
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"eng={state.engagement:.2f}  "
                    f"bor={state.boredom:.2f}  "
                    f"con={state.confusion:.2f}  "
                    f"fru={state.frustration:.2f}  "
                    f"face={state.face_detected}  "
                    f"→ {state.dominant()}"
                )
            except queue.Empty:
                pass

            if engine._stop_event.is_set():
                break
    except KeyboardInterrupt:
        pass

    engine.stop()
    print("Done.")
