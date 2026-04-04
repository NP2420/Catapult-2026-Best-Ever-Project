from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .config import AppConfig
from .models import MoodLabel, TrackSummary


logger = logging.getLogger(__name__)


SPOTIFY_SCOPES = (
    "user-read-recently-played "
    "user-read-playback-state "
    "user-modify-playback-state "
    "streaming"
)


@dataclass(slots=True)
class SpotifySnapshot:
    current_track: TrackSummary | None
    queue: list[str]
    last_refresh: datetime | None = None


class SpotifyController:
    def __init__(self, recommendation_limit: int = 4) -> None:
        self.recommendation_limit = recommendation_limit
        self.client = self._build_client()
        self.snapshot = SpotifySnapshot(current_track=None, queue=self._fallback_tracks())

    @classmethod
    def from_config(cls, config: AppConfig) -> "SpotifyController":
        return cls(recommendation_limit=config.spotify_recommendation_limit)

    def refresh_for_mood(self, mood: MoodLabel, limit: int = 4) -> SpotifySnapshot:
        limit = limit or self.recommendation_limit
        if self.client is None:
            logger.info("Spotify client unavailable, using fallback queue for mood=%s", mood.value)
            self.snapshot = SpotifySnapshot(
                current_track=self.snapshot.current_track,
                queue=self._fallback_tracks(mood, limit),
                last_refresh=datetime.now(),
            )
            return self.snapshot

        try:
            # seeds = self._recent_seed_tracks()
            # recommendations = self.client.recommendations(
            #     seed_tracks=seeds[:5],
            #     limit=limit,
            #     **self._target_audio_profile(mood),
            # )
            # queue = [self._track_from_spotify(item) for item in recommendations.get("tracks", [])]
            self.snapshot.queue = self.get_queue_tracks()
            if not self.snapshot.queue:
                self.snapshot.queue = self._fallback_tracks(mood, limit)
            logger.info(
                "Refreshed Spotify queue for mood=%s with %d tracks",
                mood.value,
                len(self.snapshot.queue),
            )
            return self.snapshot
        except Exception:
            logger.exception("Spotify refresh failed, falling back to local queue for mood=%s", mood.value)
            self.snapshot = SpotifySnapshot(
                current_track=self.snapshot.current_track,
                queue=self._fallback_tracks(mood, limit),
                last_refresh=datetime.now(),
            )
            return self.snapshot

    def current_playback(self) -> TrackSummary | None:
        if self.client is None:
            return None

        try:
            playback = self.client.current_playback()
            item = (playback or {}).get("item")
            if not item:
                logger.debug("No active Spotify playback found")
                return None
            return self._track_from_spotify(item)
        except Exception:
            logger.exception("Failed to fetch current Spotify playback")
            return None
        
    def get_queue_tracks(self) -> list[TrackSummary]:
        resp = self.client.queue()
        tracks = resp.get("queue", [])
        return [self._track_from_spotify(track) for track in tracks]
    
    def queue_top_track(self) -> None:
        if self.client is None or not self.snapshot.queue:
            return

        try:
            self.client.add_to_queue(self.snapshot.queue[0].uri)
            self.client.next_track()
            logger.info("Queued and skipped to recommended track: %s", self.snapshot.queue[0].name)
        except Exception:
            logger.exception("Failed to queue or skip to the recommended Spotify track")
            return

    def _build_client(self) -> spotipy.Spotify | None:
        client_id = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")
        redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

        if not client_id or not client_secret:
            logger.warning("Spotify credentials missing; running in fallback music mode")
            return None

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SPOTIFY_SCOPES,
            cache_path=os.getenv("SPOTIFY_CACHE_PATH", ".spotify-study-buddy-cache"),
        )
        logger.info("Spotify client configured with OAuth redirect URI %s", redirect_uri)
        return spotipy.Spotify(auth_manager=auth_manager)

    def _recent_seed_tracks(self) -> list[str]:
        if self.client is None:
            return []

        recent = self.client.current_user_recently_played(limit=10)
        ids: list[str] = []
        for item in recent.get("items", []):
            track = item.get("track") or {}
            track_id = track.get("id")
            if track_id and track_id not in ids:
                ids.append(track_id)
        return ids or ["4cOdK2wGLETKBW3PvgPWqT"]

    def _target_audio_profile(self, mood: MoodLabel) -> dict[str, float]:
        if mood in {MoodLabel.TIRED, MoodLabel.BORED}:
            return {"target_energy": 0.8, "target_valence": 0.7, "target_tempo": 128}
        if mood is MoodLabel.FRUSTRATED:
            return {"target_energy": 0.45, "target_valence": 0.55, "target_tempo": 95}
        return {"target_energy": 0.6, "target_valence": 0.6, "target_tempo": 112}

    def _track_from_spotify(self, item: dict[str, Any]) -> TrackSummary:
        artists = item.get("artists") or [{"name": "Unknown Artist"}]
        return TrackSummary(
            track_id=item.get("id", "unknown-track"),
            name=item.get("name", "Unknown Track"),
            artist=", ".join(artist.get("name", "Unknown Artist") for artist in artists),
            uri=item.get("uri", ""),
        )

    def _fallback_tracks(self, mood: MoodLabel = MoodLabel.FOCUSED, limit: int = 4) -> list[TrackSummary]:
        defaults = {
            MoodLabel.FOCUSED: [
                ("fallback-focus-1", "Deep Focus Loop", "Study Buddy"),
                ("fallback-focus-2", "Quiet Momentum", "Study Buddy"),
                ("fallback-focus-3", "Steady Pixels", "Study Buddy"),
                ("fallback-focus-4", "Flow State", "Study Buddy"),
            ],
            MoodLabel.TIRED: [
                ("fallback-tired-1", "Wake Up Sprint", "Study Buddy"),
                ("fallback-tired-2", "Bright Notes", "Study Buddy"),
                ("fallback-tired-3", "Second Wind", "Study Buddy"),
                ("fallback-tired-4", "Keep Moving", "Study Buddy"),
            ],
            MoodLabel.BORED: [
                ("fallback-bored-1", "Color Burst", "Study Buddy"),
                ("fallback-bored-2", "Fresh Tab", "Study Buddy"),
                ("fallback-bored-3", "Level Up", "Study Buddy"),
                ("fallback-bored-4", "New Loop", "Study Buddy"),
            ],
            MoodLabel.FRUSTRATED: [
                ("fallback-frustrated-1", "Reset Breath", "Study Buddy"),
                ("fallback-frustrated-2", "Soft Landing", "Study Buddy"),
                ("fallback-frustrated-3", "Clear Head", "Study Buddy"),
                ("fallback-frustrated-4", "Gentle Progress", "Study Buddy"),
            ],
        }
        return [
            TrackSummary(track_id=track_id, name=name, artist=artist, uri=f"spotify:track:{track_id}")
            for track_id, name, artist in defaults[mood][:limit]
        ]
