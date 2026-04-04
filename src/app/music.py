from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.app.behavior import ProductivityEngine
    from src.app.models import SessionSnapshot
    from src.app.mood import WebcamMoodMonitor
    from src.app.spotify_client import SpotifyController
    from src.app.ui import BuddyWindow
else:
    from .behavior import ProductivityEngine
    from .models import SessionSnapshot
    from .mood import WebcamMoodMonitor
    from .spotify_client import SpotifyController
    from .ui import BuddyWindow


class StudyBuddyController(QObject):
    def __init__(self, window: BuddyWindow) -> None:
        super().__init__()
        self.window = window
        self.mood_monitor = WebcamMoodMonitor(interval_seconds=1.0)
        self.spotify = SpotifyController()
        self.productivity = ProductivityEngine()
        self.last_tick = datetime.now()
        self.last_queue_refresh: datetime | None = None
        self.queue_refresh_interval = timedelta(seconds=45)

        self.window.resume_requested.connect(self.productivity.resume)

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)
        self.tick_timer.start(1000)

    def start(self) -> None:
        self.mood_monitor.start()
        snapshot = self.spotify.refresh_for_mood(self.mood_monitor.latest_sample().prediction.mood)
        self.last_queue_refresh = snapshot.last_refresh
        self.tick()

    def stop(self) -> None:
        self.mood_monitor.stop()

    def tick(self) -> None:
        now = datetime.now()
        elapsed = max(1.0, (now - self.last_tick).total_seconds())
        self.last_tick = now

        sample = self.mood_monitor.latest_sample()
        decision = self.productivity.tick(sample.prediction.mood, elapsed)

        needs_refresh = decision.should_refresh_queue or self._queue_refresh_due(now)
        if needs_refresh and not self.productivity.break_state.active:
            spotify_snapshot = self.spotify.refresh_for_mood(sample.prediction.mood)
            self.last_queue_refresh = spotify_snapshot.last_refresh
            if decision.should_switch_song:
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

    app = QApplication(sys.argv)
    app.setApplicationName("AI Study Buddy")

    window = BuddyWindow()
    controller = StudyBuddyController(window)
    app.aboutToQuit.connect(controller.stop)

    window.show()
    controller.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
