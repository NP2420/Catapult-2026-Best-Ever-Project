from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .config import AppConfig
from .models import MoodLabel, TrackSummary


logger = logging.getLogger(__name__)


SPOTIFY_SCOPES = (
    "user-read-recently-played "
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-top-read "
    "streaming"
)
RECCOBEATS_AUDIO_FEATURES_URL = "https://api.reccobeats.com/v1/audio-features"
SEARCH_LIMIT_PER_QUERY = 10
RECCOBEATS_BATCH_SIZE = 8
RECCOBEATS_TIMEOUT_SECONDS = 10

MOOD_GENRES: dict[MoodLabel, list[str]] = {
    MoodLabel.FOCUSED: ["study", "ambient", "chill", "classical"],
    MoodLabel.TIRED: ["pop", "dance", "electronic", "house"],
    MoodLabel.BORED: ["indie", "rock", "electropop", "alt-rock"],
    MoodLabel.FRUSTRATED: ["acoustic", "piano", "ambient", "chill"],
}


@dataclass(slots=True)
class SpotifySnapshot:
    current_track: TrackSummary | None
    queue: list[TrackSummary]
    last_refresh: datetime | None = None


@dataclass(slots=True)
class TasteProfile:
    recent_track_ids: list[str] = field(default_factory=list)
    top_track_ids: list[str] = field(default_factory=list)
    top_artist_ids: list[str] = field(default_factory=list)
    top_artist_names: list[str] = field(default_factory=list)

    @property
    def excluded_track_ids(self) -> set[str]:
        return set(self.recent_track_ids) | set(self.top_track_ids)

    @property
    def seed_artist_names(self) -> list[str]:
        return list(dict.fromkeys(name for name in self.top_artist_names if name))


class SpotifyController:
    def __init__(self, recommendation_limit: int = 4) -> None:
        self.recommendation_limit = recommendation_limit
        self.client = self._build_client()
        self.snapshot = SpotifySnapshot(current_track=None, queue=self._fallback_tracks())
        self._profile_logged = False
        self._feature_cache: dict[str, dict[str, float] | None] = {}

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
            profile = self._build_taste_profile()
            if not self._profile_logged:
                logger.info(
                    "Initialized taste profile with %d top artists from the past month and %d recent tracks",
                    len(profile.top_artist_names),
                    len(profile.recent_track_ids),
                )
                self._profile_logged = True

            candidates = self._search_candidate_tracks(mood, profile)
            ranked_queue = self._rank_candidates(mood, candidates, profile, limit)
            if not ranked_queue:
                logger.warning("Search/rerank produced no candidates for mood=%s", mood.value)
                ranked_queue = self._fallback_tracks(mood, limit)

            self.snapshot = SpotifySnapshot(
                current_track=self.current_playback(),
                queue=ranked_queue,
                last_refresh=datetime.now(),
            )
            logger.info(
                "Built mood queue for mood=%s with %d tracks from %d candidates",
                mood.value,
                len(self.snapshot.queue),
                len(candidates),
            )
            return self.snapshot
        except Exception:
            logger.exception("Spotify search/rerank refresh failed for mood=%s", mood.value)
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

    def queue_top_track(self) -> None:
        if self.client is None or not self.snapshot.queue:
            return

        try:
            self.client.add_to_queue(self.snapshot.queue[0].uri)
            self.client.next_track()
            logger.info("Queued and skipped to recommended track: %s", self.snapshot.queue[0].name)
        except Exception:
            logger.exception("Failed to queue or skip to the recommended Spotify track")

    def apply_queue_to_spotify(self, max_tracks: int | None = None) -> int:
        if self.client is None or not self.snapshot.queue:
            return 0

        desired_tracks = self.snapshot.queue[: max_tracks or len(self.snapshot.queue)]
        existing_ids = {track.track_id for track in self.get_queue_tracks()}
        current_track = self.current_playback()
        if current_track is not None:
            existing_ids.add(current_track.track_id)

        queued = 0
        for track in desired_tracks:
            if track.track_id in existing_ids:
                logger.debug("Skipping already-present Spotify queue track: %s", track.name)
                continue
            try:
                self.client.add_to_queue(track.uri)
                existing_ids.add(track.track_id)
                queued += 1
                logger.info("Added mood-queue track to Spotify queue: %s", track.name)
            except Exception:
                logger.exception("Failed to add recommended track to Spotify queue: %s", track.name)

        logger.info("Applied %d tracks from the mood queue to Spotify", queued)
        return queued

    def get_queue_tracks(self) -> list[TrackSummary]:
        if self.client is None:
            return []

        try:
            response = self.client.queue()
        except Exception:
            logger.exception("Failed to fetch current Spotify queue")
            return []

        tracks = response.get("queue", [])
        logger.debug("Fetched %d tracks from the current Spotify queue", len(tracks))
        return [self._track_from_spotify(track) for track in tracks]

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

    def _build_taste_profile(self) -> TasteProfile:
        if self.client is None:
            return TasteProfile()

        recent = self.client.current_user_recently_played(limit=15)
        top_tracks = self.client.current_user_top_tracks(limit=10, time_range="short_term")
        top_artists = self.client.current_user_top_artists(limit=10, time_range="short_term")

        profile = TasteProfile()

        for item in recent.get("items", []):
            track = item.get("track") or {}
            track_id = track.get("id")
            if track_id and track_id not in profile.recent_track_ids:
                profile.recent_track_ids.append(track_id)

        for track in top_tracks.get("items", []):
            track_id = track.get("id")
            if track_id and track_id not in profile.top_track_ids:
                profile.top_track_ids.append(track_id)

        for artist in top_artists.get("items", []):
            artist_id = artist.get("id")
            artist_name = artist.get("name")
            if artist_id and artist_id not in profile.top_artist_ids:
                profile.top_artist_ids.append(artist_id)
            if artist_name and artist_name not in profile.top_artist_names:
                profile.top_artist_names.append(artist_name)

        logger.debug(
            "Built taste profile: %d recent tracks, %d top tracks, %d top artists",
            len(profile.recent_track_ids),
            len(profile.top_track_ids),
            len(profile.top_artist_names),
        )
        return profile

    def _search_candidate_tracks(self, mood: MoodLabel, profile: TasteProfile) -> list[TrackSummary]:
        if self.client is None:
            return []

        queries = self._candidate_queries_for_mood(mood, profile)
        candidates: list[TrackSummary] = []
        seen_ids: set[str] = set()
        excluded = profile.excluded_track_ids
        current_track = self.current_playback()
        if current_track:
            excluded.add(current_track.track_id)

        for query in queries:
            try:
                logger.debug("Spotify search query for mood=%s: %s", mood.value, query)
                response = self.client.search(q=query, type="track", limit=SEARCH_LIMIT_PER_QUERY, market="US")
            except Exception:
                logger.warning("Spotify search failed for query=%s", query, exc_info=True)
                continue

            items = ((response or {}).get("tracks") or {}).get("items") or []
            for item in items:
                track = self._track_from_spotify(item)
                if track.track_id in excluded or track.track_id in seen_ids:
                    continue
                seen_ids.add(track.track_id)
                candidates.append(track)

        logger.info("Collected %d candidate tracks for mood=%s", len(candidates), mood.value)
        return candidates

    def _candidate_queries_for_mood(self, mood: MoodLabel, profile: TasteProfile) -> list[str]:
        artist_names = profile.seed_artist_names[:4]
        genres = MOOD_GENRES[mood][:3]

        queries: list[str] = []
        for genre in genres:
            queries.append(f"genre:{genre}")
        for artist_name in artist_names:
            queries.append(f'artist:"{artist_name}"')
        for artist_name in artist_names[:2]:
            for genre in genres[:2]:
                queries.append(f'artist:"{artist_name}" genre:{genre}')

        return list(dict.fromkeys(queries))

    def _rank_candidates(
        self,
        mood: MoodLabel,
        candidates: list[TrackSummary],
        profile: TasteProfile,
        limit: int,
    ) -> list[TrackSummary]:
        if not candidates:
            return []

        feature_map = self._fetch_audio_features_with_retry([track.track_id for track in candidates])
        ranked: list[tuple[float, TrackSummary]] = []

        for track in candidates:
            score = self._score_track(mood, track, feature_map.get(track.track_id), profile)
            if score is None:
                continue
            features = feature_map.get(track.track_id)
            if features:
                track.energy = features.get("energy", track.energy)
                track.tempo = features.get("tempo", track.tempo)
                track.valence = features.get("valence", track.valence)
            ranked.append((score, track))

        ranked.sort(key=lambda item: item[0], reverse=True)
        logger.info("Ranked %d candidate tracks for mood=%s", len(ranked), mood.value)
        return [track for _, track in ranked[:limit]]

    def _fetch_audio_features_with_retry(self, track_ids: list[str]) -> dict[str, dict[str, float] | None]:
        unique_ids = [track_id for track_id in dict.fromkeys(track_ids) if track_id]
        result: dict[str, dict[str, float] | None] = {}
        missing = [track_id for track_id in unique_ids if track_id not in self._feature_cache]

        for start in range(0, len(missing), RECCOBEATS_BATCH_SIZE):
            batch = missing[start : start + RECCOBEATS_BATCH_SIZE]
            batch_features = self._fetch_reccobeats_batch(batch)
            for track_id in batch:
                self._feature_cache[track_id] = batch_features.get(track_id)

        for track_id in unique_ids:
            feature = self._feature_cache.get(track_id)
            if feature is None and track_id in self._feature_cache:
                feature = self._retry_reccobeats_single(track_id)
                self._feature_cache[track_id] = feature
            result[track_id] = self._feature_cache.get(track_id)

        return result

    def _fetch_reccobeats_batch(self, track_ids: list[str]) -> dict[str, dict[str, float] | None]:
        if not track_ids:
            return {}

        try:
            response = requests.get(
                RECCOBEATS_AUDIO_FEATURES_URL,
                params={"ids": ",".join(track_ids)},
                headers={"Accept": "application/json"},
                timeout=RECCOBEATS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.warning("Reccobeats batch request failed for %d track ids", len(track_ids), exc_info=True)
            return {track_id: None for track_id in track_ids}

        parsed = self._parse_reccobeats_payload(payload)
        logger.debug("Reccobeats returned features for %d/%d tracks", len(parsed), len(track_ids))
        return {track_id: parsed.get(track_id) for track_id in track_ids}

    def _retry_reccobeats_single(self, track_id: str) -> dict[str, float] | None:
        try:
            response = requests.get(
                RECCOBEATS_AUDIO_FEATURES_URL,
                params={"ids": track_id},
                headers={"Accept": "application/json"},
                timeout=RECCOBEATS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.debug("Reccobeats single retry failed for track_id=%s", track_id, exc_info=True)
            return None

        parsed = self._parse_reccobeats_payload(payload)
        feature = parsed.get(track_id)
        if feature is None:
            logger.debug("Reccobeats returned no matching feature payload for track_id=%s", track_id)
        return feature

    def _parse_reccobeats_payload(self, payload: dict[str, Any]) -> dict[str, dict[str, float]]:
        parsed: dict[str, dict[str, float]] = {}
        for item in payload.get("content", []):
            href = item.get("href", "")
            track_id = href.rstrip("/").rsplit("/", 1)[-1] if href else ""
            if not track_id:
                continue
            parsed[track_id] = {
                "energy": float(item.get("energy", 0.5)),
                "tempo": float(item.get("tempo", 110.0)),
                "valence": float(item.get("valence", 0.5)),
            }
        return parsed

    def _score_track(
        self,
        mood: MoodLabel,
        track: TrackSummary,
        features: dict[str, float] | None,
        profile: TasteProfile,
    ) -> float | None:
        if features is None:
            logger.debug("Skipping track %s because audio features were unavailable", track.track_id)
            return None

        target = self._target_audio_profile(mood)
        energy_score = 1.0 - abs(features["energy"] - target["energy"])
        valence_score = 1.0 - abs(features["valence"] - target["valence"])
        tempo_delta = abs(features["tempo"] - target["tempo"]) / 80.0
        tempo_score = max(0.0, 1.0 - tempo_delta)

        artist_match = 0.15 if any(name in track.artist for name in profile.seed_artist_names[:5]) else 0.0
        freshness_bonus = 0.1 if track.track_id not in profile.excluded_track_ids else -0.5

        return (energy_score * 0.4) + (valence_score * 0.3) + (tempo_score * 0.3) + artist_match + freshness_bonus

    def _target_audio_profile(self, mood: MoodLabel) -> dict[str, float]:
        if mood in {MoodLabel.TIRED, MoodLabel.BORED}:
            return {"energy": 0.8, "valence": 0.7, "tempo": 128.0}
        if mood is MoodLabel.FRUSTRATED:
            return {"energy": 0.45, "valence": 0.55, "tempo": 95.0}
        return {"energy": 0.6, "valence": 0.6, "tempo": 112.0}

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
