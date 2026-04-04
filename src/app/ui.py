from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPixmap
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


MOOD_COLORS = {
    MoodLabel.FOCUSED: QColor("#3FB67B"),
    MoodLabel.TIRED: QColor("#F2C14E"),
    MoodLabel.BORED: QColor("#5DADE2"),
    MoodLabel.FRUSTRATED: QColor("#E76F51"),
}


class PixelBuddyWidget(QLabel):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._mood = MoodLabel.FOCUSED
        self._frame_index = 0
        self._frames = self._build_frames(self._mood)
        self.setFixedSize(96, 96)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._timer.start(300)
        self._render()

    def set_mood(self, mood: MoodLabel) -> None:
        if mood == self._mood:
            return
        self._mood = mood
        self._frame_index = 0
        self._frames = self._build_frames(mood)
        self._render()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)

    def _advance_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self._render()

    def _render(self) -> None:
        self.setPixmap(self._frames[self._frame_index])

    def _build_frames(self, mood: MoodLabel) -> list[QPixmap]:
        offsets = [-1, 1, -2, 0]
        return [self._draw_buddy(mood, body_offset=value) for value in offsets]

    def _draw_buddy(self, mood: MoodLabel, body_offset: int) -> QPixmap:
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(QRect(12, 10, 72, 72), QColor("#132026"))

        color = MOOD_COLORS[mood]
        painter.fillRect(QRect(24, 22 + body_offset, 48, 40), color)
        painter.fillRect(QRect(18, 30 + body_offset, 12, 16), color)
        painter.fillRect(QRect(66, 30 + body_offset, 12, 16), color)
        painter.fillRect(QRect(30, 60 + body_offset, 10, 12), color)
        painter.fillRect(QRect(56, 60 + body_offset, 10, 12), color)

        eye_y = 34 + body_offset
        painter.fillRect(QRect(34, eye_y, 6, 6), QColor("#F8F9FA"))
        painter.fillRect(QRect(56, eye_y, 6, 6), QColor("#F8F9FA"))

        pupil_y = eye_y + (2 if mood is MoodLabel.TIRED else 1)
        painter.fillRect(QRect(36, pupil_y, 2, 2), QColor("#0B090A"))
        painter.fillRect(QRect(58, pupil_y, 2, 2), QColor("#0B090A"))

        pen = QPen(QColor("#0B090A"))
        pen.setWidth(2)
        painter.setPen(pen)

        if mood is MoodLabel.FOCUSED:
            painter.drawLine(39, 50 + body_offset, 57, 50 + body_offset)
        elif mood is MoodLabel.TIRED:
            painter.drawLine(39, 51 + body_offset, 57, 49 + body_offset)
        elif mood is MoodLabel.BORED:
            painter.drawLine(39, 52 + body_offset, 57, 52 + body_offset)
        else:
            painter.drawLine(39, 52 + body_offset, 48, 48 + body_offset)
            painter.drawLine(48, 48 + body_offset, 57, 52 + body_offset)

        painter.end()
        return pixmap


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
