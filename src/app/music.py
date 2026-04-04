from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

# supports both relative and absolute imports when running as a script or module
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.app.behavior import ProductivityEngine
    from src.app.config import AppConfig, load_app_config
    from src.app.logging_config import configure_logging
    from src.app.models import SessionSnapshot
    from src.app.mood import WebcamMoodMonitor
    from src.app.spotify_client import SpotifyController
    from src.app.ui import BuddyWindow
else:
    from .behavior import ProductivityEngine
    from .config import AppConfig, load_app_config
    from .logging_config import configure_logging
    from .models import SessionSnapshot
    from .mood import WebcamMoodMonitor
    from .spotify_client import SpotifyController
    from .ui import BuddyWindow


logger = logging.getLogger(__name__)


class StudyBuddyController(QObject):
    def __init__(self, window: BuddyWindow, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.window = window
        self.mood_monitor = WebcamMoodMonitor.from_config(config)
        self.spotify = SpotifyController.from_config(config)
        self.productivity = ProductivityEngine(
            short_term_threshold_seconds=config.short_term_threshold_seconds,
            long_term_threshold_seconds=config.long_term_threshold_seconds,
            minimum_break_seconds=config.minimum_break_seconds,
        )
        self.last_tick = datetime.now()
        self.last_queue_refresh: datetime | None = None
        self.queue_refresh_interval = timedelta(seconds=config.queue_refresh_seconds)

        self.window.resume_requested.connect(self.productivity.resume)

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)
        self.tick_timer.start(config.controller_tick_interval_ms)

    def start(self) -> None:
        logger.info("Starting Study Buddy controller")
        self.mood_monitor.start()
        initial_mood = self.mood_monitor.latest_sample().prediction.mood
        snapshot = self.spotify.refresh_for_mood(
            initial_mood,
            limit=self.config.spotify_recommendation_limit,
        )
        applied_count = self.spotify.apply_queue_to_spotify(
            max_tracks=self.config.spotify_recommendation_limit,
        )
        logger.info("Applied %d startup mood-queue tracks to Spotify", applied_count)
        self.last_queue_refresh = snapshot.last_refresh
        self.tick()

    def stop(self) -> None:
        logger.info("Stopping Study Buddy controller")
        self.mood_monitor.stop()

    def tick(self) -> None:
        now = datetime.now()
        elapsed = max(1.0, (now - self.last_tick).total_seconds())
        self.last_tick = now

        sample = self.mood_monitor.latest_sample()
        decision = self.productivity.tick(sample.prediction.mood, elapsed)

        needs_refresh = decision.should_refresh_queue or self._queue_refresh_due(now)
        if needs_refresh and not self.productivity.break_state.active:
            logger.debug(
                "Refreshing queue for mood=%s, switch_song=%s",
                sample.prediction.mood.value,
                decision.should_switch_song,
            )
            spotify_snapshot = self.spotify.refresh_for_mood(
                sample.prediction.mood,
                limit=self.config.spotify_recommendation_limit,
            )
            applied_count = self.spotify.apply_queue_to_spotify(
                max_tracks=self.config.spotify_recommendation_limit,
            )
            logger.info(
                "Applied %d refreshed mood-queue tracks to Spotify for mood=%s",
                applied_count,
                sample.prediction.mood.value,
            )
            self.last_queue_refresh = spotify_snapshot.last_refresh
            if decision.should_switch_song:
                logger.info("Short-term fatigue reached; switching song")
                self.spotify.queue_top_track()

        current_track = self.spotify.current_playback() or self.spotify.snapshot.current_track
        snapshot = SessionSnapshot(
            mood=sample.prediction.mood,
            confidence=sample.prediction.confidence,
            fatigue_seconds=self.productivity.fatigue_seconds,
            break_state=self.productivity.break_state,
            current_track=current_track,
            upcoming_tracks=self.spotify.snapshot.queue,
            last_queue_refresh=self.last_queue_refresh,
        )
        self.window.update_snapshot(snapshot, webcam_available=sample.available)

    def _queue_refresh_due(self, now: datetime) -> bool:
        if self.last_queue_refresh is None:
            return True
        return (now - self.last_queue_refresh) >= self.queue_refresh_interval


def main() -> int:
    load_dotenv()
    config = load_app_config()
    configure_logging(config)
    logger.info("Loaded app config: %s", config)

    app = QApplication(sys.argv)
    app.setApplicationName("AI Study Buddy")

    window = BuddyWindow(config)
    controller = StudyBuddyController(window, config)
    app.aboutToQuit.connect(controller.stop)

    window.show()
    controller.start()
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
