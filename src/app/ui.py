from __future__ import annotations

import logging
import random
from enum import StrEnum
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig
from .models import BreakState, SessionSnapshot, TrackSummary, fatigue_state_from_score


logger = logging.getLogger(__name__)


STATE_MESSAGES = {
    "awake": [
        "Keep up the focus, you're doing great!",
        "Eyes on the prize!",
        "Almost there, stay sharp!",
    ],
    "drowsy": [
        "Take a deep breath...",
        "Maybe a quick stretch would help!",
        "Stay hydrated, don't slump!",
    ],
    "tired": [
        "Time to reset your energy a bit.",
        "A quick movement break could help here.",
        "Let's wake things back up.",
    ],
    "break": [
        "Time for a break! Relax a bit.",
        "Step away and refresh your mind.",
        "Stretch or grab a snack!",
    ],
    "default": [
        "You're doing great, keep it up!",
        "Almost there, keep going!",
        "Don't forget to blink!",
    ],
}


BUBBLE_VISIBLE_MS = 5_000
BUBBLE_INTERVAL_MS = 1 * 60 * 1_000
VISIBLE_QUEUE_ITEMS = 1


class FocusLevel(StrEnum):
    HAPPY = "happy"
    TIRED = "tired"
    SLEEPY = "sleepy"
    BREAK = "break"


class SpeechBubbleLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                background: #FFFFFF;
                color: #111827;
                border-radius: 10px;
                border: 2px solid #E5E7EB;
                padding: 8px 12px;
                font-size: 10px;
                font-family: Consolas, monospace;
            }
            """
        )

        self._tail = QLabel(self)
        self._tail.setFixedSize(12, 12)
        self._tail.setStyleSheet(
            """
            QLabel {
                background: #FFFFFF;
                border-right: 2px solid #E5E7EB;
                border-bottom: 2px solid #E5E7EB;
                padding: 0px;
            }
            """
        )
        self._tail.lower()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._tail.move(
            (self.width() - self._tail.width()) // 2,
            self.height() - self._tail.height() // 2,
        )


class PixelBuddyWidget(QLabel):
    clicked = Signal()

    def __init__(self, frame_root: Path, size_px: int, frame_interval_ms: int) -> None:
        super().__init__()
        self._frame_root = frame_root
        self._ema_score = 0.2
        self._focus_level = focus_level_for_score(self._ema_score)
        self._frame_index = 0
        self._frames = self._load_frames(self._focus_level)
        self.setFixedSize(size_px, size_px)
        self.setScaledContents(False)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._timer.start(frame_interval_ms)
        self._render()

    def set_state(self, ema_score: float, break_state: BreakState | None = None) -> None:
        next_focus_level = focus_level_for_snapshot(ema_score, break_state)
        if next_focus_level == self._focus_level:
            self._ema_score = ema_score
            return
        self._ema_score = ema_score
        self._focus_level = next_focus_level
        self._frame_index = 0
        self._frames = self._load_frames(self._focus_level)
        self._render()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def _advance_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self._render()

    def _render(self) -> None:
        frame = self._frames[self._frame_index]
        self.setPixmap(
            frame.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def _load_frames(self, focus_level: FocusLevel) -> list[QPixmap]:
        frame_dir = self._frame_root / focus_level.value
        frame_paths = sorted(path for path in frame_dir.glob("*.png") if path.is_file())
        frames = [QPixmap(str(path)) for path in frame_paths]
        valid_frames = [frame for frame in frames if not frame.isNull()]
        if valid_frames:
            logger.debug("Loaded %d buddy frames for focus level %s", len(valid_frames), focus_level.value)
            return valid_frames

        logger.warning("No valid buddy frames found in %s", frame_dir)
        placeholder = QPixmap(self.size())
        placeholder.fill(Qt.GlobalColor.transparent)
        return [placeholder]


def focus_level_for_score(ema_score: float) -> FocusLevel:
    state_label = fatigue_state_from_score(ema_score)
    if state_label == "awake":
        return FocusLevel.HAPPY
    if state_label == "drowsy":
        return FocusLevel.TIRED
    return FocusLevel.SLEEPY


def focus_level_for_snapshot(ema_score: float, break_state: BreakState | None = None) -> FocusLevel:
    if break_state and break_state.active and break_state.seconds_remaining > 0:
        return FocusLevel.BREAK
    return focus_level_for_score(ema_score)


class BuddyWindow(QWidget):
    resume_requested = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._expanded = False
        self._drag_origin: QPoint | None = None
        self._did_drag = False
        self._bottom_right: QPoint | None = None
        self._last_message: str | None = None
        self._latest_snapshot: SessionSnapshot | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._bubble = SpeechBubbleLabel()
        self._bubble.setFixedWidth(config.buddy_size_px + 36)
        self._bubble.setVisible(False)

        self._buddy = PixelBuddyWidget(
            frame_root=config.buddy_frame_root,
            size_px=config.buddy_size_px,
            frame_interval_ms=config.buddy_frame_interval_ms,
        )
        self._buddy.clicked.connect(self.toggle_expanded)

        buddy_column = QVBoxLayout()
        buddy_column.setContentsMargins(0, 0, 0, 0)
        buddy_column.setSpacing(4)
        buddy_column.addWidget(self._bubble, alignment=Qt.AlignmentFlag.AlignHCenter)
        buddy_column.addWidget(self._buddy, alignment=Qt.AlignmentFlag.AlignBottom)

        buddy_container = QWidget()
        buddy_container.setLayout(buddy_column)
        buddy_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        buddy_container.setMinimumSize(
            config.buddy_size_px + 36,
            config.buddy_size_px + 80,
        )

        self._panel = QFrame()
        self._panel.setStyleSheet(
            """
            QFrame {
                background: rgba(17, 24, 39, 230);
                border-radius: 16px;
                border: none;
            }
            QLabel {
                background: transparent;
                color: #F8FAFC;
            }
            QPushButton {
                background: #F2C14E;
                color: #111827;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: #6B7280;
                color: #D1D5DB;
            }
            """
        )
        self._panel.setVisible(False)

        title = QLabel("AI Study Buddy")
        title.setFont(QFont("Consolas", 12, QFont.Weight.Bold))

        self._quit_button = QPushButton("✕")
        self._quit_button.setFixedSize(24, 24)
        self._quit_button.setStyleSheet(
            """
            QPushButton {
                background: #EF4444;
                color: #FFFFFF;
                border: none;
                border-radius: 12px;
                padding: 0px;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #DC2626;
            }
            """
        )
        self._quit_button.clicked.connect(QApplication.quit)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self._quit_button)

        self._song_label = QLabel("Playing: Waiting for playback")
        self._queue_label = QLabel("Up next: building queue...")
        self._score_label = QLabel("Tiredness: 0.20 (awake)")
        self._status_label = QLabel("Camera: connecting")
        self._break_label = QLabel("Break: not active")
        self._last_update_label = QLabel("Queue refresh: pending")
        self._resume_button = QPushButton("Resume")
        self._resume_button.setEnabled(False)
        self._resume_button.clicked.connect(self.resume_requested.emit)
        self._reset_position_button = QPushButton("Reset Position")
        self._reset_position_button.clicked.connect(self._reset_position)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(8)
        panel_layout.addLayout(title_row)
        panel_layout.addWidget(self._song_label)
        panel_layout.addWidget(self._queue_label)
        panel_layout.addWidget(self._score_label)
        panel_layout.addWidget(self._status_label)
        panel_layout.addWidget(self._break_label)
        panel_layout.addWidget(self._last_update_label)
        panel_layout.addWidget(self._resume_button)
        panel_layout.addWidget(self._reset_position_button)
        self._panel.setLayout(panel_layout)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._panel)
        layout.addWidget(buddy_container, alignment=Qt.AlignmentFlag.AlignBottom)
        self.setLayout(layout)

        self._message_timer = QTimer(self)
        self._message_timer.timeout.connect(self._show_state_message)
        self._message_timer.start(BUBBLE_INTERVAL_MS)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_bubble)

        self._refresh_layout()

    def _choose_message(self, snapshot: SessionSnapshot) -> str:
        if snapshot.break_state.active:
            choices = STATE_MESSAGES["break"]
        else:
            if not snapshot.face_detected:
                choices = STATE_MESSAGES["default"]
            else:
                state_label = fatigue_state_from_score(snapshot.ema_score)
                choices = STATE_MESSAGES.get(state_label, STATE_MESSAGES["default"])

        choices = [message for message in choices if message != self._last_message]
        message = random.choice(choices) if choices else random.choice(STATE_MESSAGES["default"])
        self._last_message = message
        return message

    def _show_state_message(self) -> None:
        if self._latest_snapshot is None:
            return
        message = self._choose_message(self._latest_snapshot)
        self._bubble.setText(message)
        self._bubble.setVisible(True)
        self._refresh_layout()
        self._hide_timer.start(BUBBLE_VISIBLE_MS)
        logger.debug("Buddy message shown: %s", message)

    def _hide_bubble(self) -> None:
        if not self._bubble.isVisible():
            return
        self._bubble.setVisible(False)
        self._refresh_layout()

    def toggle_expanded(self) -> None:
        if self._did_drag:
            return
        self._expanded = not self._expanded
        self._panel.setVisible(self._expanded)
        self._refresh_layout()

    def _reset_position(self) -> None:
        self._bottom_right = None
        self._refresh_layout()

    def position_bottom_right(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        margin = self._config.window_margin_px
        self._bottom_right = QPoint(
            geometry.right() - margin,
            geometry.bottom() - margin,
        )
        self.move(
            self._bottom_right.x() - self.width(),
            self._bottom_right.y() - self.height(),
        )

    def update_snapshot(self, snapshot: SessionSnapshot, webcam_available: bool) -> None:
        self._latest_snapshot = snapshot
        self._buddy.set_state(snapshot.ema_score, snapshot.break_state)
        self._score_label.setText(
            f"Tiredness: {snapshot.ema_score:.2f} ({snapshot.state_label}) | 5s avg: {snapshot.rolling_score:.2f}"
        )
        self._status_label.setText(
            "Camera: "
            + (
                f"{'live' if webcam_available else 'offline, using fallback'}"
                f" | {'face detected' if snapshot.face_detected else 'no face detected'}"
                f" | Fatigue: {snapshot.fatigue_seconds:.0f}s"
            )
        )
        self._song_label.setText(
            "Playing: "
            + (
                f"{snapshot.current_track.name} - {snapshot.current_track.artist}"
                if snapshot.current_track
                else ("Spotify disabled (demo mode)" if not snapshot.spotify_enabled else "No active Spotify playback")
            )
        )
        self._queue_label.setText(
            f"Up next: {self._format_upcoming_tracks(snapshot.current_track, snapshot.upcoming_tracks)}"
        )
        self._update_break(snapshot.break_state)
        self._last_update_label.setText(
            "Last queue refresh: "
            + (
                snapshot.last_queue_refresh.strftime("%I:%M:%S %p")
                if snapshot.last_queue_refresh
                else "pending"
            )
        )

        if snapshot.break_state.active and not self._bubble.isVisible():
            self._show_state_message()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._did_drag = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_origin
            if not self._did_drag:
                delta = event.globalPosition().toPoint() - (self.frameGeometry().topLeft() + self._drag_origin)
                if delta.manhattanLength() > 4:
                    self._did_drag = True
            self.move(new_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None and self._did_drag:
            screen = self.screen() or QApplication.primaryScreen()
            geo = self.frameGeometry()
            br_x = geo.x() + geo.width()
            br_y = geo.y() + geo.height()
            if screen is not None:
                avail = screen.availableGeometry()
                br_x = max(avail.left() + self.width(), min(br_x, avail.right()))
                br_y = max(avail.top() + self.height(), min(br_y, avail.bottom()))
            self._bottom_right = QPoint(br_x, br_y)
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowDeactivate:
            self.activateWindow()
        return super().event(event)

    def _refresh_size(self) -> None:
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

    def _refresh_layout(self) -> None:
        self._refresh_size()
        if self._bottom_right is not None:
            x = self._bottom_right.x() - self.width()
            y = self._bottom_right.y() - self.height()
            clamped = self._clamp_to_screen(x, y)
            self.move(clamped)
        else:
            self.position_bottom_right()

    def _clamp_to_screen(self, x: int, y: int) -> QPoint:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return QPoint(x, y)
        geo = screen.availableGeometry()
        x = max(geo.left(), min(x, geo.right() - self.width()))
        y = max(geo.top(), min(y, geo.bottom() - self.height()))
        return QPoint(x, y)

    def _update_break(self, break_state: BreakState) -> None:
        if not break_state.active:
            self._break_label.setText("Break: not active")
            self._resume_button.setEnabled(False)
            return

        minutes, seconds = divmod(max(0, break_state.seconds_remaining), 60)
        label = f"Break: {minutes:02d}:{seconds:02d} remaining"
        if break_state.can_resume:
            label = "Break: complete, ready to resume"
        self._break_label.setText(label)
        self._resume_button.setEnabled(break_state.can_resume)

    def _format_upcoming_tracks(
        self,
        current_track: TrackSummary | None,
        upcoming_tracks: list[TrackSummary],
    ) -> str:
        visible_tracks: list[str] = []
        current_track_id = current_track.track_id if current_track else None

        for track in upcoming_tracks:
            if current_track_id and track.track_id == current_track_id:
                continue
            visible_tracks.append(f"{track.name} - {track.artist}")
            if len(visible_tracks) >= VISIBLE_QUEUE_ITEMS:
                break

        return ", ".join(visible_tracks) or "No queued recommendations"
