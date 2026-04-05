from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import AppConfig
from .models import MoodPrediction, clamp_score


logger = logging.getLogger(__name__)


YOLO_CLASSES = ["closed_eye", "closed_mouth", "open_eye", "open_mouth"]
EYE_WEIGHT = 0.7
MOUTH_WEIGHT = 0.3
EMA_ALPHA = 0.15
WINDOW_SECS = 5.0
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_IOU_THRESHOLD = 0.45
DEFAULT_EMA_SCORE = 0.2
YOLO_CHECKPOINT = Path(
    os.getenv(
        "STUDY_BUDDY_YOLO_CKPT",
        str(Path(__file__).resolve().parents[1] / "ml-focus" / "best.pt"),
    )
)

_INFERENCE_LOCK = threading.Lock()
_MODEL: YOLO | None = None
_MODEL_LOAD_FAILED = False
_EMA_SCORE = DEFAULT_EMA_SCORE
_SCORE_HISTORY: deque[tuple[float, float]] = deque()


def predict_mood_from_image(frame: np.ndarray) -> MoodPrediction:
    """
    Run the realtime tiredness pipeline on a single webcam frame.

    The caller already provides the raw frame, so the entire YOLO inference,
    tiredness scoring, EMA smoothing, and rolling-average update happen here.
    """

    global _MODEL, _MODEL_LOAD_FAILED, _EMA_SCORE

    with _INFERENCE_LOCK:
        if _MODEL is None and not _MODEL_LOAD_FAILED:
            if not YOLO_CHECKPOINT.exists():
                logger.error("YOLO checkpoint not found at %s", YOLO_CHECKPOINT)
                _MODEL_LOAD_FAILED = True
            else:
                logger.info("Loading realtime tiredness model from %s", YOLO_CHECKPOINT)
                _MODEL = YOLO(str(YOLO_CHECKPOINT))
                _MODEL.fuse()

        raw_score = DEFAULT_EMA_SCORE if _MODEL is None else 0.0
        face_detected = False

        if _MODEL is not None:
            try:
                results = _MODEL.predict(
                    frame,
                    conf=YOLO_CONFIDENCE_THRESHOLD,
                    iou=YOLO_IOU_THRESHOLD,
                    verbose=False,
                    stream=False,
                )[0]

                boxes = results.boxes
                face_detected = boxes is not None and len(boxes) > 0

                if face_detected:
                    class_names = [YOLO_CLASSES[int(class_index)] for class_index in boxes.cls]
                    closed_eyes = class_names.count("closed_eye")
                    open_eyes = class_names.count("open_eye")
                    open_mouths = class_names.count("open_mouth")
                    closed_mouths = class_names.count("closed_mouth")

                    eye_score = closed_eyes / max(closed_eyes + open_eyes, 1)
                    mouth_score = open_mouths / max(open_mouths + closed_mouths, 1)
                    raw_score = (EYE_WEIGHT * eye_score) + (MOUTH_WEIGHT * mouth_score)
            except Exception:
                logger.exception("Realtime YOLO inference failed; returning fallback tiredness score")
                raw_score = DEFAULT_EMA_SCORE
                face_detected = False

        _EMA_SCORE = clamp_score((EMA_ALPHA * raw_score) + ((1.0 - EMA_ALPHA) * _EMA_SCORE))

        now = time.time()
        _SCORE_HISTORY.append((now, _EMA_SCORE))
        while _SCORE_HISTORY and _SCORE_HISTORY[0][0] < now - WINDOW_SECS:
            _SCORE_HISTORY.popleft()

        rolling_score = (
            sum(score for _, score in _SCORE_HISTORY) / len(_SCORE_HISTORY)
            if _SCORE_HISTORY
            else _EMA_SCORE
        )

        return MoodPrediction(
            raw_score=raw_score,
            ema_score=_EMA_SCORE,
            rolling_score=rolling_score,
            face_detected=face_detected,
        )


@dataclass(slots=True)
class WebcamSample:
    frame: np.ndarray | None
    prediction: MoodPrediction
    available: bool


class WebcamInferenceMonitor:
    """
    Capture webcam frames and publish realtime tiredness predictions.

    The prediction function owns the entire model pipeline, so this monitor
    only handles camera I/O, threading, and the latest sample state.
    """

    def __init__(
        self,
        camera_index: int = 0,
        interval_seconds: float = 1.0,
        smoothing_window: int = 10,
    ) -> None:
        self.camera_index = camera_index
        self.interval_seconds = interval_seconds
        self.smoothing_window = max(1, smoothing_window)
        self._latest_sample = WebcamSample(
            frame=None,
            prediction=MoodPrediction(
                raw_score=DEFAULT_EMA_SCORE,
                ema_score=DEFAULT_EMA_SCORE,
                rolling_score=DEFAULT_EMA_SCORE,
                face_detected=False,
            ),
            available=False,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_availability: bool | None = None
        self._last_state_label: str | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "WebcamInferenceMonitor":
        return cls(
            camera_index=config.camera_index,
            interval_seconds=config.mood_poll_interval_seconds,
            smoothing_window=config.mood_smoothing_window,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        logger.info(
            "Starting webcam inference monitor",
            extra={
                "camera_index": self.camera_index,
                "interval_seconds": self.interval_seconds,
            },
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="webcam-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("Stopping webcam inference monitor")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def latest_sample(self) -> WebcamSample:
        with self._lock:
            return self._latest_sample

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        try:
            if not capture.isOpened():
                logger.warning("Webcam could not be opened, using fallback predictions")

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    fallback = self._fallback_prediction()
                    self._update_sample(None, fallback, available=False)
                    time.sleep(self.interval_seconds)
                    continue

                prediction = predict_mood_from_image(frame)
                self._update_sample(frame, prediction, available=True)
                time.sleep(self.interval_seconds)
        finally:
            capture.release()
            logger.info("Webcam capture released")

    def _update_sample(self, frame: np.ndarray | None, prediction: MoodPrediction, available: bool) -> None:
        if available != self._last_availability:
            logger.info("Webcam availability changed to %s", "online" if available else "offline")
            self._last_availability = available

        if prediction.state_label != self._last_state_label:
            logger.debug(
                "Realtime tiredness state changed to %s (ema=%.2f)",
                prediction.state_label,
                prediction.ema_score,
            )
            self._last_state_label = prediction.state_label

        with self._lock:
            self._latest_sample = WebcamSample(
                frame=frame,
                prediction=prediction,
                available=available,
            )

    def _fallback_prediction(self) -> MoodPrediction:
        return MoodPrediction(
            raw_score=DEFAULT_EMA_SCORE,
            ema_score=DEFAULT_EMA_SCORE,
            rolling_score=DEFAULT_EMA_SCORE,
            face_detected=False,
        )


WebcamMoodMonitor = WebcamInferenceMonitor
