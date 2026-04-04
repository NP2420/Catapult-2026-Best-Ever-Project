from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class MoodLabel(StrEnum):
    FOCUSED = "focused"
    TIRED = "tired"
    BORED = "bored"
    FRUSTRATED = "frustrated"


@dataclass(slots=True)
class MoodPrediction:
    mood: MoodLabel
    confidence: float
    probabilities: dict[MoodLabel, float]
    captured_at: datetime = field(default_factory=datetime.now)


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
    mood: MoodLabel
    confidence: float
    fatigue_seconds: float
    break_state: BreakState
    current_track: TrackSummary | None
    upcoming_tracks: list[TrackSummary]
    last_queue_refresh: datetime | None
