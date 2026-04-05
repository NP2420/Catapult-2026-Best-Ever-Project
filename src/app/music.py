from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import os
import sys
import signal
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
    from src.app.models import SessionSnapshot, fatigue_state_from_score
    from src.app.mood import WebcamInferenceMonitor
    from src.app.spotify_client import SpotifyController
    from src.app.ui import BuddyWindow
else:
    from .behavior import ProductivityEngine
    from .config import AppConfig, load_app_config
    from .logging_config import configure_logging
    from .models import SessionSnapshot, fatigue_state_from_score
    from .mood import WebcamInferenceMonitor
    from .spotify_client import SpotifyController
    from .ui import BuddyWindow


logger = logging.getLogger(__name__)


def parse_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the AI Study Buddy desktop app.",
    )
    parser.add_argument(
        "--spotify",
        dest="spotify_enabled",
        action="store_true",
        default=True,
        help="Enable live Spotify integration (default).",
    )
    parser.add_argument(
        "--no-spotify",
        dest="spotify_enabled",
        action="store_false",
        help="Disable Spotify OAuth/API calls and run in offline demo/debug mode.",
    )
    return parser.parse_known_args(argv)


class StudyBuddyController(QObject):
    def __init__(self, window: BuddyWindow, config: AppConfig, spotify_enabled: bool = True) -> None:
        super().__init__()
        self.config = config
        self.window = window
        self.mood_monitor = WebcamInferenceMonitor.from_config(config)
        self.spotify = SpotifyController.from_config(config, spotify_enabled=spotify_enabled)
        self.productivity = ProductivityEngine(
            short_term_threshold_seconds=config.short_term_threshold_seconds,
            long_term_threshold_seconds=config.long_term_threshold_seconds,
            minimum_break_seconds=config.minimum_break_seconds,
        )
        self.last_tick = datetime.now()
        self.last_queue_refresh: datetime | None = None
        self.queue_refresh_interval = timedelta(seconds=config.queue_refresh_seconds)
        self.minimum_queue_refresh_interval = timedelta(seconds=config.minimum_queue_refresh_seconds)
        self._refresh_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spotify-refresh")
        self._pending_refresh: Future[tuple[datetime | None, int, bool]] | None = None

        self.window.resume_requested.connect(self.productivity.resume)

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)
        self.tick_timer.start(config.controller_tick_interval_ms)

    def start(self) -> None:
        logger.info("Starting Study Buddy controller")
        self.mood_monitor.start()
        initial_score = self.mood_monitor.latest_sample().prediction.ema_score
        self._schedule_refresh(initial_score, should_switch_song=False, reason="startup")
        self.tick()

    def stop(self) -> None:
        logger.info("Stopping Study Buddy controller")
        self.mood_monitor.stop()
        self._refresh_executor.shutdown(wait=False, cancel_futures=True)

    def tick(self) -> None:
        self._collect_completed_refresh()

        now = datetime.now()
        elapsed = max(1.0, (now - self.last_tick).total_seconds())
        self.last_tick = now

        sample = self.mood_monitor.latest_sample()
        decision = self.productivity.tick(sample.prediction.ema_score, elapsed)

        needs_refresh = decision.should_refresh_queue or self._queue_refresh_due(now)
        if needs_refresh and not self.productivity.break_state.active and self._queue_refresh_allowed(now):
            state_label = fatigue_state_from_score(sample.prediction.ema_score)
            logger.debug(
                "Refreshing queue for score=%.2f (%s), switch_song=%s",
                sample.prediction.ema_score,
                state_label,
                decision.should_switch_song,
            )
            self._schedule_refresh(
                sample.prediction.ema_score,
                should_switch_song=decision.should_switch_song,
                reason="fatigue trigger" if decision.should_switch_song else "scheduled refresh",
            )
        elif needs_refresh and not self.productivity.break_state.active:
            logger.debug(
                "Skipping Spotify refresh because minimum interval of %.0fs has not elapsed",
                self.minimum_queue_refresh_interval.total_seconds(),
            )

        spotify_snapshot = self.spotify.get_snapshot()
        snapshot = SessionSnapshot(
            raw_score=sample.prediction.raw_score,
            ema_score=sample.prediction.ema_score,
            rolling_score=sample.prediction.rolling_score,
            face_detected=sample.prediction.face_detected,
            fatigue_seconds=self.productivity.fatigue_seconds,
            break_state=self.productivity.break_state,
            current_track=spotify_snapshot.current_track,
            upcoming_tracks=spotify_snapshot.queue,
            last_queue_refresh=spotify_snapshot.last_refresh or self.last_queue_refresh,
            spotify_enabled=self.spotify.spotify_enabled,
        )
        self.window.update_snapshot(snapshot, webcam_available=sample.available)

    def _queue_refresh_due(self, now: datetime) -> bool:
        if self.last_queue_refresh is None:
            return True
        return (now - self.last_queue_refresh) >= self.queue_refresh_interval

    def _queue_refresh_allowed(self, now: datetime) -> bool:
        if self.last_queue_refresh is None:
            return True
        return (now - self.last_queue_refresh) >= self.minimum_queue_refresh_interval

    def _schedule_refresh(self, ema_score: float, should_switch_song: bool, reason: str) -> None:
        if self._pending_refresh and not self._pending_refresh.done():
            logger.debug("Skipping Spotify refresh schedule because one is already running")
            return

        logger.info(
            "Scheduling Spotify refresh for score=%.2f (%s, %s)",
            ema_score,
            fatigue_state_from_score(ema_score),
            reason,
        )
        self._pending_refresh = self._refresh_executor.submit(
            self._run_refresh_job,
            ema_score,
            should_switch_song,
        )

    def _run_refresh_job(self, ema_score: float, should_switch_song: bool) -> tuple[datetime | None, int, bool]:
        snapshot = self.spotify.refresh_for_score(
            ema_score,
            limit=self.config.spotify_recommendation_limit,
        )
        applied_count = self.spotify.apply_queue_to_spotify(
            max_tracks=self.config.spotify_recommendation_limit,
        )
        if should_switch_song:
            self.spotify.queue_top_track()
        return snapshot.last_refresh, applied_count, should_switch_song

    def _collect_completed_refresh(self) -> None:
        if self._pending_refresh is None or not self._pending_refresh.done():
            return

        try:
            last_refresh, applied_count, should_switch_song = self._pending_refresh.result()
            self.last_queue_refresh = last_refresh
            logger.info(
                "Completed Spotify refresh: applied=%d, switched_song=%s",
                applied_count,
                should_switch_song,
            )
        except Exception:
            logger.exception("Spotify refresh worker failed")
        finally:
            self._pending_refresh = None


def main() -> int:
    args, qt_args = parse_cli_args(sys.argv[1:])

    load_dotenv()
    config = load_app_config()
    configure_logging(config)
    logger.info("Loaded app config: %s", config)
    logger.info("Spotify integration enabled: %s", args.spotify_enabled)

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("AI Study Buddy")

    window = BuddyWindow(config)
    controller = StudyBuddyController(window, config, spotify_enabled=args.spotify_enabled)
    app.aboutToQuit.connect(controller.stop)

    window.show()
    controller.start()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    exit_code = app.exec()
    controller.stop()
    logger.info("Exiting with code %d", exit_code)
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
