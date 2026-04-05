from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class AppConfig:
    """
    Centralized configuration for the Study Buddy application, loaded from environment variables with defaults.

    - base_dir: The base directory of the application, used for resolving relative paths.
    - buddy_frame_root: Directory containing the Pixel Buddy animation frames.
    - camera_index: Index of the webcam to use for mood detection.
    - mood_poll_interval_seconds: How often to poll the webcam for mood updates.
    - mood_smoothing_window: Number of recent mood samples to smooth over for stability.
    - controller_tick_interval_ms: How often the main controller should update (in milliseconds).
    - queue_refresh_seconds: How often to attempt the periodic Spotify queue refresh.
    - minimum_queue_refresh_seconds: Minimum time between any Spotify queue refreshes.
    - spotify_recommendation_limit: Number of tracks to request from Spotify for each mood refresh.
    - short_term_threshold_seconds: Fatigue threshold for short-term breaks.
    - long_term_threshold_seconds: Fatigue threshold for long-term breaks.
    - minimum_break_seconds: Minimum duration for any break.
    - buddy_frame_interval_ms: How often to advance the Pixel Buddy animation frames (in milliseconds).
    - buddy_size_px: Size of the Pixel Buddy widget in pixels.
    - window_margin_px: Margin around the Buddy window in pixels.
    - log_level: Logging level for the application (e.g., DEBUG, INFO, WARNING).
    - log_format: Format string for log messages.
    """
    
    base_dir: Path
    buddy_frame_root: Path
    camera_index: int
    mood_poll_interval_seconds: float
    mood_smoothing_window: int
    controller_tick_interval_ms: int
    queue_refresh_seconds: int
    minimum_queue_refresh_seconds: int
    spotify_recommendation_limit: int
    short_term_threshold_seconds: int
    long_term_threshold_seconds: int
    minimum_break_seconds: int
    buddy_frame_interval_ms: int
    buddy_size_px: int
    window_margin_px: int
    log_level: str
    log_format: str


def load_app_config() -> AppConfig:
    """
    Load the application configuration from environment variables and provide default values.
    """
    
    base_dir = Path(__file__).resolve().parents[2]
    return AppConfig(
        base_dir=base_dir,
        buddy_frame_root=base_dir / "src" / "buddy-img",
        camera_index=_env_int("STUDY_BUDDY_CAMERA_INDEX", 0),
        mood_poll_interval_seconds=_env_float("STUDY_BUDDY_MOOD_POLL_SECONDS", 1.0),
        mood_smoothing_window=_env_int("STUDY_BUDDY_MOOD_SMOOTHING_WINDOW", 10),
        controller_tick_interval_ms=_env_int("STUDY_BUDDY_TICK_MS", 1000),
        queue_refresh_seconds=_env_int("STUDY_BUDDY_QUEUE_REFRESH_SECONDS", 45),
        minimum_queue_refresh_seconds=_env_int("STUDY_BUDDY_MIN_QUEUE_REFRESH_SECONDS", 15),
        spotify_recommendation_limit=_env_int("STUDY_BUDDY_SPOTIFY_LIMIT", 1),
        short_term_threshold_seconds=_env_int("STUDY_BUDDY_SHORT_TERM_THRESHOLD", 15),
        long_term_threshold_seconds=_env_int("STUDY_BUDDY_LONG_TERM_THRESHOLD", 60),
        minimum_break_seconds=_env_int("STUDY_BUDDY_MIN_BREAK_SECONDS", 300),
        buddy_frame_interval_ms=_env_int("STUDY_BUDDY_FRAME_INTERVAL_MS", 300),
        buddy_size_px=_env_int("STUDY_BUDDY_BUDDY_SIZE_PX", 96),
        window_margin_px=_env_int("STUDY_BUDDY_WINDOW_MARGIN_PX", 24),
        log_level=os.getenv("STUDY_BUDDY_LOG_LEVEL", "INFO").upper(),
        log_format=os.getenv(
            "STUDY_BUDDY_LOG_FORMAT",
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        ),
    )
