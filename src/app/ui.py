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
from .models import BreakState, MoodLabel, SessionSnapshot




logger = logging.getLogger(__name__)


STATE_MESSAGES = {
    "focus": [
        "Keep up the focus, you're doing great!",
        "Eyes on the prize!",
        "Almost there, stay sharp!"
    ],
    "tired": [
        "Take a deep breath…",
        "Maybe a quick stretch would help!",
        "Stay hydrated, don't slump!"
    ],
    "break": [
        "Time for a break! Relax a bit",
        "Step away and refresh your mind",
        "Stretch or grab a snack!"
    ],
    "default": [
        "You're doing great, keep it up!",
        "Almost there, keep going!",
        "Don't forget to blink!"
    ],
}


# How long the bubble stays visible before auto-hiding
BUBBLE_VISIBLE_MS = 8_000
# How often a new message appears (5 minutes)
BUBBLE_INTERVAL_MS = 1 * 5 * 1_000




class FocusLevel(StrEnum):
    HAPPY = "happy"
    TIRED = "tired"
    SLEEPY = "sleepy"




FOCUS_LEVEL_BY_MOOD = {
    MoodLabel.FOCUSED: FocusLevel.HAPPY,
    MoodLabel.TIRED: FocusLevel.TIRED,
    MoodLabel.BORED: FocusLevel.TIRED,
    MoodLabel.FRUSTRATED: FocusLevel.SLEEPY,
}




class SpeechBubbleLabel(QLabel):
    """A styled speech bubble with a downward-pointing tail."""


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
        # Tail is a small rotated square below the bubble, drawn as a child label
        self._tail = QLabel(self)
        self._tail.setFixedSize(12, 12)
        self._tail.setStyleSheet(
            """
            QLabel {
                background: #FFFFFF;
                border-right: 2px solid #E5E7EB;
                border-bottom: 2px solid #E5E7EB;
                border-radius: 0px;
                padding: 0px;
            }
            """
        )
        self._tail.lower()  # render behind bubble text


    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Position the tail: centred horizontally, peeking below the bubble
        tail = self._tail
        tail.move(
            (self.width() - tail.width()) // 2,
            self.height() - tail.height() // 2,
        )
        # Rotate 45° via CSS transform so it looks like a downward pointer
        tail.setStyleSheet(
            """
            QLabel {
                background: #FFFFFF;
                border-right: 2px solid #E5E7EB;
                border-bottom: 2px solid #E5E7EB;
                border-radius: 0px;
                padding: 0px;
                transform: rotate(45deg);
            }
            """
        )




class PixelBuddyWidget(QLabel):
    clicked = Signal()


    def __init__(self, frame_root: Path, size_px: int, frame_interval_ms: int) -> None:
        super().__init__()
        self._frame_root = frame_root
        self._mood = MoodLabel.FOCUSED
        self._focus_level = focus_level_for_mood(self._mood)
        self._frame_index = 0
        self._frames = self._load_frames(self._focus_level)
        self.setFixedSize(size_px, size_px)
        self.setScaledContents(False)


        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._timer.start(frame_interval_ms)
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
        super().mousePressEvent(event)


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
            logger.debug(
                "Loaded %d buddy frames for focus level %s",
                len(valid_frames),
                focus_level.value,
            )
            return valid_frames


        logger.warning("No valid buddy frames found in %s", frame_dir)
        placeholder = QPixmap(self.size())
        placeholder.fill(Qt.GlobalColor.transparent)
        return [placeholder]




def focus_level_for_mood(mood: MoodLabel) -> FocusLevel:
    return FOCUS_LEVEL_BY_MOOD.get(mood, FocusLevel.TIRED)




class BuddyWindow(QWidget):
    resume_requested = Signal()


    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._expanded = False
        self._drag_origin: QPoint | None = None
        self._did_drag: bool = False
        self._last_message: str | None = None  # avoid repeating the same message twice


        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


        # --- buddy + speech bubble, stacked vertically ---
        self._bubble = SpeechBubbleLabel()
        self._bubble.setFixedWidth(config.buddy_size_px + 20)
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


        # --- info panel ---
        self._panel = QFrame()
        self._panel.setStyleSheet(
            """
            QFrame {
                background: rgba(17, 24, 39, 50);
                border-radius: 16px;
                border: none;
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


        self._song_label = QLabel("Playing: Waiting for playback")
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
        layout.addWidget(buddy_container, alignment=Qt.AlignmentFlag.AlignBottom)
        self.setLayout(layout)


        # --- speech bubble timers ---
        # Fires every 5 minutes to show a new message
        self._message_timer = QTimer(self)
        self._message_timer.timeout.connect(self._show_state_message)
        self._message_timer.start(BUBBLE_INTERVAL_MS)


        # Single-shot timer that hides the bubble after BUBBLE_VISIBLE_MS
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_bubble)


        self._refresh_size()
        self.position_bottom_right()


    # Speech bubble helpers
    def _choose_message(self, snapshot: SessionSnapshot) -> str:
        """Return a message depending on mood or break state."""
        if snapshot.break_state.active:
            choices = STATE_MESSAGES["break"]
        else:
            focus_level = focus_level_for_mood(snapshot.mood)
            if focus_level == FocusLevel.HAPPY:
                choices = STATE_MESSAGES["focus"]
            elif focus_level == FocusLevel.TIRED:
                choices = STATE_MESSAGES["tired"]
            else:
                choices = STATE_MESSAGES["default"]
       
        # Avoid repeating the last message
        choices = [m for m in choices if m != self._last_message]
        message = random.choice(choices) if choices else random.choice(STATE_MESSAGES["default"])
        self._last_message = message
        return message
   
    def _show_state_message(self) -> None:
        # snapshot must be passed or stored somewhere
        if not hasattr(self, "_latest_snapshot"):
            return
        message = self._choose_message(self._latest_snapshot)
        self._bubble.setText(message)
        self._bubble.setVisible(True)
        self._refresh_size()
        self._hide_timer.start(BUBBLE_VISIBLE_MS)
        logger.debug("Buddy message shown: %s", message)


    def _hide_bubble(self) -> None:
        self._bubble.setVisible(False)
        self._refresh_size()


    # Existing methods (unchanged)


    def toggle_expanded(self) -> None:
        if self._did_drag:
            return
        self._expanded = not self._expanded
        self._panel.setVisible(self._expanded)
        self._refresh_size()


    def position_bottom_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        margin = self._config.window_margin_px
        self.move(
            geometry.right() - self.width() - margin,
            geometry.bottom() - self.height() - margin,
        )


    def update_snapshot(self, snapshot: SessionSnapshot, webcam_available: bool) -> None:
        self._latest_snapshot = snapshot  # store for message logic
        self._buddy.set_mood(snapshot.mood)
        self._mood_label.setText(
            f"Mood: {snapshot.mood.value} ({snapshot.confidence:.0%} confidence)"
        )
        self._status_label.setText(
            f"Camera: {'live' if webcam_available else 'offline, using fallback'} | Fatigue: {snapshot.fatigue_seconds:.0f}s"
        )
        self._song_label.setText(
            "Playing: "
            + (
                f"{snapshot.current_track.name} - {snapshot.current_track.artist}"
                if snapshot.current_track
                else "No active Spotify playback"
            )
        )
        queue_text = ", ".join(
            f"{track.name} - {track.artist}" for track in snapshot.upcoming_tracks[:1]
        )
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
            self._did_drag = False
        super().mousePressEvent(event)


    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_origin
            if not self._did_drag:
                # Only count as a drag once the cursor moves more than 4px
                delta = event.globalPosition().toPoint() - (self.frameGeometry().topLeft() + self._drag_origin)
                if delta.manhattanLength() > 4:
                    self._did_drag = True
            self.move(new_pos)
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

