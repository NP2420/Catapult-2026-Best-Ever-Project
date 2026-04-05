from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


DROWSY_SCORE_THRESHOLD = 0.35
TIRED_SCORE_THRESHOLD = 0.65


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def fatigue_state_from_score(score: float) -> str:
    score = clamp_score(score)
    if score < DROWSY_SCORE_THRESHOLD:
        return "awake"
    if score < TIRED_SCORE_THRESHOLD:
        return "drowsy"
    return "tired"


@dataclass(slots=True)
class MoodPrediction:
    raw_score: float
    ema_score: float
    rolling_score: float
    face_detected: bool
    captured_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.raw_score = clamp_score(self.raw_score)
        self.ema_score = clamp_score(self.ema_score)
        self.rolling_score = clamp_score(self.rolling_score)

    @property
    def state_label(self) -> str:
        return fatigue_state_from_score(self.ema_score)


@dataclass(slots=True)
class TrackSummary:
    track_id: str
    name: str
    artist: str
    uri: str
    energy: float = 0.5
    tempo: float = 110.0
    valence: float = 0.5


@dataclass(slots=True)
class BreakState:
    active: bool = False
    seconds_remaining: int = 0
    can_resume: bool = False


@dataclass(slots=True)
class SessionSnapshot:
    raw_score: float
    ema_score: float
    rolling_score: float
    face_detected: bool
    fatigue_seconds: float
    break_state: BreakState
    current_track: TrackSummary | None
    upcoming_tracks: list[TrackSummary]
    last_queue_refresh: datetime | None

    @property
    def state_label(self) -> str:
        return fatigue_state_from_score(self.ema_score)
