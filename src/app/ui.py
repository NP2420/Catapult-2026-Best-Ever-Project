from __future__ import annotations

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

from .models import BreakState, MoodLabel, SessionSnapshot


class FocusLevel(StrEnum):
    HAPPY = "happy"
    TIRED = "tired"
    SLEEPY = "sleepy"


FRAME_ROOT = Path(__file__).resolve().parents[1] / "buddy-img"
FOCUS_LEVEL_BY_MOOD = {
    MoodLabel.FOCUSED: FocusLevel.HAPPY,
    MoodLabel.TIRED: FocusLevel.TIRED,
    MoodLabel.BORED: FocusLevel.TIRED,
    MoodLabel.FRUSTRATED: FocusLevel.SLEEPY,
}


class PixelBuddyWidget(QLabel):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._mood = MoodLabel.FOCUSED
        self._focus_level = focus_level_for_mood(self._mood)
        self._frame_index = 0
        self._frames = self._load_frames(self._focus_level)
        self.setFixedSize(96, 96)
        self.setScaledContents(False)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._timer.start(300)
        self._render()

    def set_mood(self, mood: MoodLabel) -> None:
        next_focus_level = focus_level_for_mood(mood)
        if mood == self._mood and next_focus_level == self._focus_level:
            return
        self._mood = mood
        self._focus_level = next_focus_level
        self._frame_index = 0
        self._frames = self._load_frames(self._focus_level)
        self._render()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)

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
        frame_dir = FRAME_ROOT / focus_level.value
        frame_paths = sorted(path for path in frame_dir.glob("*.png") if path.is_file())
        frames = [QPixmap(str(path)) for path in frame_paths]
        valid_frames = [frame for frame in frames if not frame.isNull()]
        if valid_frames:
            return valid_frames

        placeholder = QPixmap(self.size())
        placeholder.fill(Qt.GlobalColor.transparent)
        return [placeholder]


def focus_level_for_mood(mood: MoodLabel) -> FocusLevel:
    return FOCUS_LEVEL_BY_MOOD.get(mood, FocusLevel.TIRED)


class BuddyWindow(QWidget):
    resume_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._expanded = False
        self._drag_origin: QPoint | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._buddy = PixelBuddyWidget()
        self._buddy.clicked.connect(self.toggle_expanded)

        self._panel = QFrame()
        self._panel.setStyleSheet(
            """
            QFrame {
                background: rgba(17, 24, 39, 235);
                border: 2px solid rgba(255, 255, 255, 35);
                border-radius: 16px;
            }
            QLabel {
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

        self._song_label = QLabel("Current song: Waiting for playback")
        self._queue_label = QLabel("Up next: building queue...")
        self._mood_label = QLabel("Mood: focused")
        self._status_label = QLabel("Camera: connecting")
        self._break_label = QLabel("Break: not active")
        self._last_update_label = QLabel("Queue refresh: pending")
        self._resume_button = QPushButton("Resume")
        self._resume_button.setEnabled(False)
        self._resume_button.clicked.connect(self.resume_requested.emit)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(8)
        panel_layout.addWidget(title)
        panel_layout.addWidget(self._song_label)
        panel_layout.addWidget(self._queue_label)
        panel_layout.addWidget(self._mood_label)
        panel_layout.addWidget(self._status_label)
        panel_layout.addWidget(self._break_label)
        panel_layout.addWidget(self._last_update_label)
        panel_layout.addWidget(self._resume_button)
        self._panel.setLayout(panel_layout)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._panel)
        layout.addWidget(self._buddy, alignment=Qt.AlignmentFlag.AlignBottom)
        self.setLayout(layout)
        self._refresh_size()
        self.position_bottom_right()

    def toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._panel.setVisible(self._expanded)
        self._refresh_size()
        self.position_bottom_right()

    def position_bottom_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        margin = 24
        self.move(
            geometry.right() - self.width() - margin,
            geometry.bottom() - self.height() - margin,
        )

    def update_snapshot(self, snapshot: SessionSnapshot, webcam_available: bool) -> None:
        self._buddy.set_mood(snapshot.mood)
        self._mood_label.setText(
            f"Mood: {snapshot.mood.value} ({snapshot.confidence:.0%} confidence)"
        )
        self._status_label.setText(
            f"Camera: {'live' if webcam_available else 'offline, using fallback'} | Fatigue: {snapshot.fatigue_seconds:.0f}s"
        )
        self._song_label.setText(
            "Current song: "
            + (
                f"{snapshot.current_track.name} - {snapshot.current_track.artist}"
                if snapshot.current_track
                else "No active Spotify playback"
            )
        )
        queue_text = ", ".join(f"{track.name} - {track.artist}" for track in snapshot.upcoming_tracks[:4])
        self._queue_label.setText(f"Up next: {queue_text or 'No queued recommendations'}")
        self._update_break(snapshot.break_state)
        self._last_update_label.setText(
            "Queue refresh: "
            + (
                snapshot.last_queue_refresh.strftime("%I:%M:%S %p")
                if snapshot.last_queue_refresh
                else "pending"
            )
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowDeactivate:
            self.activateWindow()
        return super().event(event)

    def _refresh_size(self) -> None:
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

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
