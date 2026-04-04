from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .models import MoodLabel, MoodPrediction


def predict_mood_from_image(frame: np.ndarray) -> MoodPrediction:
    """Dummy seam for the future model.

    Replace the heuristic block with a real model call that returns the same
    MoodPrediction shape. The rest of the app only depends on this function.
    """

    resized = cv2.resize(frame, (96, 96))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray)) / 255.0
    motion = float(np.std(gray)) / 255.0

    probabilities = {
        MoodLabel.FOCUSED: max(0.05, 0.55 + (motion * 0.2) - abs(brightness - 0.5)),
        MoodLabel.TIRED: max(0.05, 0.55 - brightness + 0.15),
        MoodLabel.BORED: max(0.05, 0.4 + (0.18 - motion)),
        MoodLabel.FRUSTRATED: max(0.05, 0.25 + abs(brightness - 0.5) + (motion * 0.15)),
    }

    total = sum(probabilities.values())
    normalized = {mood: value / total for mood, value in probabilities.items()}
    best_mood = max(normalized, key=normalized.get)
    confidence = normalized[best_mood]

    return MoodPrediction(
        mood=best_mood,
        confidence=confidence,
        probabilities=normalized,
    )


class MoodSmoother:
    def __init__(self, window_size: int = 10) -> None:
        self._window: deque[MoodPrediction] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, prediction: MoodPrediction) -> MoodPrediction:
        with self._lock:
            self._window.append(prediction)
            grouped: dict[MoodLabel, float] = {mood: 0.0 for mood in MoodLabel}
            for item in self._window:
                for mood, score in item.probabilities.items():
                    grouped[mood] += score

            total = sum(grouped.values()) or 1.0
            normalized = {mood: score / total for mood, score in grouped.items()}
            best_mood = max(normalized, key=normalized.get)

            return MoodPrediction(
                mood=best_mood,
                confidence=normalized[best_mood],
                probabilities=normalized,
            )


@dataclass(slots=True)
class WebcamSample:
    frame: np.ndarray | None
    prediction: MoodPrediction
    available: bool


class WebcamMoodMonitor:
    def __init__(self, camera_index: int = 0, interval_seconds: float = 1.0) -> None:
        self.camera_index = camera_index
        self.interval_seconds = interval_seconds
        self.smoother = MoodSmoother(window_size=10)
        self._latest_sample = WebcamSample(
            frame=None,
            prediction=MoodPrediction(
                mood=MoodLabel.FOCUSED,
                confidence=1.0,
                probabilities={mood: 1.0 if mood is MoodLabel.FOCUSED else 0.0 for mood in MoodLabel},
            ),
            available=False,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="webcam-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def latest_sample(self) -> WebcamSample:
        with self._lock:
            return self._latest_sample

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    fallback = self._fallback_prediction()
                    self._update_sample(None, fallback, available=False)
                    time.sleep(self.interval_seconds)
                    continue

                prediction = predict_mood_from_image(frame)
                smoothed = self.smoother.add(prediction)
                self._update_sample(frame, smoothed, available=True)
                time.sleep(self.interval_seconds)
        finally:
            capture.release()

    def _update_sample(self, frame: np.ndarray | None, prediction: MoodPrediction, available: bool) -> None:
        with self._lock:
            self._latest_sample = WebcamSample(
                frame=frame,
                prediction=prediction,
                available=available,
            )

    def _fallback_prediction(self) -> MoodPrediction:
        weights = {
            MoodLabel.FOCUSED: 0.7,
            MoodLabel.TIRED: 0.1,
            MoodLabel.BORED: 0.1,
            MoodLabel.FRUSTRATED: 0.1,
        }
        mood = random.choices(list(weights.keys()), weights=weights.values(), k=1)[0]
        return MoodPrediction(mood=mood, confidence=0.55, probabilities=weights)
