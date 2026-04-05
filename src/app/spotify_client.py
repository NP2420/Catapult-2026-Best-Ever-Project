from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .config import AppConfig
from .models import (
    DROWSY_SCORE_THRESHOLD,
    TIRED_SCORE_THRESHOLD,
    TrackSummary,
    clamp_score,
    fatigue_state_from_score,
)


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
SEARCH_OFFSET_STEP = 10
SEARCH_OFFSET_CYCLE = 3
RECENT_RECOMMENDATION_HISTORY = 80
RECENT_ARTIST_HISTORY = 80
TASTE_PROFILE_TTL_SECONDS = 600
CANDIDATE_CACHE_TTL_SECONDS = 600

AWAKE_GENRES = ["study", "ambient", "chill", "classical"]
DROWSY_GENRES = ["lofi", "indie", "electropop", "alt-rock"]
TIRED_GENRES = ["pop", "dance", "electronic", "house"]


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
    def __init__(self, recommendation_limit: int = 4, spotify_enabled: bool = True) -> None:
        self.recommendation_limit = recommendation_limit
        self.spotify_enabled = spotify_enabled
        self.client = self._build_client()
        self.snapshot = SpotifySnapshot(current_track=None, queue=self._fallback_tracks())
        self._profile_logged = False
        self._feature_cache: dict[str, dict[str, float] | None] = {}
        self._client_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._refresh_generation = 0
        self._recent_recommendation_ids: deque[str] = deque(maxlen=RECENT_RECOMMENDATION_HISTORY)
        self._recent_recommendation_artists: deque[str] = deque(maxlen=RECENT_ARTIST_HISTORY)
        self._cached_profile: TasteProfile | None = None
        self._profile_cached_at: datetime | None = None
        self._candidate_cache: dict[str, tuple[list[TrackSummary], datetime]] = {}
        self._spotipy_call_count = 0
        self._reccobeats_call_count = 0

    @classmethod
    def from_config(cls, config: AppConfig, spotify_enabled: bool = True) -> "SpotifyController":
        return cls(
            recommendation_limit=config.spotify_recommendation_limit,
            spotify_enabled=spotify_enabled,
        )

    def refresh_for_score(self, ema_score: float, limit: int = 4) -> SpotifySnapshot:
        self._spotipy_call_count = 0
        self._reccobeats_call_count = 0
        score = clamp_score(ema_score)
        state_label = fatigue_state_from_score(score)
        limit = limit or self.recommendation_limit
        if self.client is None:
            logger.info(
                "Spotify client unavailable, using fallback queue for score=%.2f (%s)",
                score,
                state_label,
            )
            snapshot = SpotifySnapshot(
                current_track=self.get_snapshot().current_track,
                queue=self._fallback_tracks(score, limit),
                last_refresh=datetime.now(),
            )
            self._set_snapshot(snapshot)
            logger.info("refresh_for_score made %d spotipy and %d reccobeats API request(s)", self._spotipy_call_count, self._reccobeats_call_count)
            return snapshot

        try:
            current_track = self.current_playback()

            profile = self._build_taste_profile()
            if not self._profile_logged:
                logger.info(
                    "Initialized taste profile with %d top artists from the past month and %d recent tracks",
                    len(profile.top_artist_names),
                    len(profile.recent_track_ids),
                )
                self._profile_logged = True

            candidates = self._search_candidate_tracks(score, profile, current_track)
            ranked_queue = self._rank_candidates(score, candidates, profile, limit)
            if not ranked_queue:
                logger.warning("Search/rerank produced no candidates for score=%.2f (%s)", score, state_label)
                ranked_queue = self._fallback_tracks(score, limit)

            snapshot = SpotifySnapshot(
                current_track=current_track,
                queue=ranked_queue,
                last_refresh=datetime.now(),
            )
            self._remember_recommendations(snapshot.queue)
            self._refresh_generation += 1
            self._set_snapshot(snapshot)
            logger.info(
                "Built score-aware queue for score=%.2f (%s) with %d tracks from %d candidates",
                score,
                state_label,
                len(snapshot.queue),
                len(candidates),
            )
            logger.info("refresh_for_score made %d spotipy and %d reccobeats API request(s)", self._spotipy_call_count, self._reccobeats_call_count)
            return snapshot
        except Exception:
            logger.exception("Spotify search/rerank refresh failed for score=%.2f (%s)", score, state_label)
            snapshot = SpotifySnapshot(
                current_track=self.get_snapshot().current_track,
                queue=self._fallback_tracks(score, limit),
                last_refresh=datetime.now(),
            )
            self._set_snapshot(snapshot)
            logger.info("refresh_for_score made %d spotipy and %d reccobeats API request(s)", self._spotipy_call_count, self._reccobeats_call_count)
            return snapshot

    def current_playback(self) -> TrackSummary | None:
        with self._client_lock:
            if self.client is None:
                return None

            try:
                self._spotipy_call_count += 1
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
        snapshot = self.get_snapshot()
        if self.client is None or not snapshot.queue:
            return

        with self._client_lock:
            try:
                self.client.add_to_queue(snapshot.queue[0].uri)
                self.client.next_track()
                logger.info("Queued and skipped to recommended track: %s", snapshot.queue[0].name)
            except Exception:
                logger.exception("Failed to queue or skip to the recommended Spotify track")

    def apply_queue_to_spotify(self, max_tracks: int | None = None) -> int:
        snapshot = self.get_snapshot()
        if self.client is None or not snapshot.queue:
            return 0

        desired_tracks = snapshot.queue[: max_tracks or len(snapshot.queue)]
        existing_ids = {track.track_id for track in self.get_queue_tracks()}
        current_track = self.current_playback()
        if current_track is not None:
            existing_ids.add(current_track.track_id)

        queued = 0
        with self._client_lock:
            for track in desired_tracks:
                if track.track_id in existing_ids:
                    logger.debug("Skipping already-present Spotify queue track: %s", track.name)
                    continue
                try:
                    self.client.add_to_queue(track.uri)
                    existing_ids.add(track.track_id)
                    queued += 1
                    logger.info("Added score-aware track to Spotify queue: %s", track.name)
                except Exception:
                    logger.exception("Failed to add recommended track to Spotify queue: %s", track.name)

        logger.info("Applied %d score-aware tracks to Spotify", queued)
        return queued

    def get_queue_tracks(self) -> list[TrackSummary]:
        with self._client_lock:
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

    def get_snapshot(self) -> SpotifySnapshot:
        with self._state_lock:
            return SpotifySnapshot(
                current_track=self.snapshot.current_track,
                queue=list(self.snapshot.queue),
                last_refresh=self.snapshot.last_refresh,
            )

    def _build_client(self) -> spotipy.Spotify | None:
        if not self.spotify_enabled:
            logger.info("Spotify integration disabled via command-line option; using fallback music mode")
            return None

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

        now = datetime.now()
        if (
            self._cached_profile is not None
            and self._profile_cached_at is not None
            and (now - self._profile_cached_at).total_seconds() < TASTE_PROFILE_TTL_SECONDS
        ):
            logger.debug("Returning cached taste profile (age=%.0fs)", (now - self._profile_cached_at).total_seconds())
            return self._cached_profile

        with self._client_lock:
            self._spotipy_call_count += 3
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
        self._cached_profile = profile
        self._profile_cached_at = datetime.now()
        return profile

    def _search_candidate_tracks(
        self, ema_score: float, profile: TasteProfile, current_track: TrackSummary | None = None,
    ) -> list[TrackSummary]:
        if self.client is None:
            return []

        score = clamp_score(ema_score)
        state_label = fatigue_state_from_score(score)

        now = datetime.now()
        cached = self._candidate_cache.get(state_label)
        if cached is not None:
            cached_candidates, cached_at = cached
            if cached_candidates and (now - cached_at).total_seconds() < CANDIDATE_CACHE_TTL_SECONDS:
                logger.info(
                    "Reusing %d cached candidates for state=%s (age=%.0fs)",
                    len(cached_candidates),
                    state_label,
                    (now - cached_at).total_seconds(),
                )
                return list(cached_candidates)

        queries = self._candidate_queries_for_score(score, profile)
        candidates: list[TrackSummary] = []
        seen_ids: set[str] = set()
        excluded = set(profile.excluded_track_ids) | set(self._recent_recommendation_ids)
        if current_track:
            excluded.add(current_track.track_id)

        for index, query in enumerate(queries):
            try:
                logger.debug("Spotify search query for score=%.2f (%s): %s", score, state_label, query)
                offset = self._query_offset(index)
                with self._client_lock:
                    self._spotipy_call_count += 1
                    response = self.client.search(
                        q=query,
                        type="track",
                        limit=SEARCH_LIMIT_PER_QUERY,
                        offset=offset,
                        market="US",
                    )
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

        logger.info("Collected %d candidate tracks for score=%.2f (%s)", len(candidates), score, state_label)

        feature_map = self._fetch_audio_features_with_retry([t.track_id for t in candidates])
        for track in candidates:
            features = feature_map.get(track.track_id)
            if features:
                track.energy = features.get("energy", track.energy)
                track.tempo = features.get("tempo", track.tempo)
                track.valence = features.get("valence", track.valence)

        self._candidate_cache[state_label] = (list(candidates), datetime.now())
        return candidates

    def _candidate_queries_for_score(self, ema_score: float, profile: TasteProfile) -> list[str]:
        artist_names = profile.seed_artist_names[:6]
        genres = self._genres_for_score(ema_score)[:4]

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
        ema_score: float,
        candidates: list[TrackSummary],
        profile: TasteProfile,
        limit: int,
    ) -> list[TrackSummary]:
        if not candidates:
            return []

        score = clamp_score(ema_score)
        state_label = fatigue_state_from_score(score)
        ranked: list[tuple[float, TrackSummary]] = []

        for track in candidates:
            features = self._feature_cache.get(track.track_id)
            ranking_score = self._score_track(score, track, features, profile)
            if ranking_score is None:
                continue
            ranked.append((ranking_score, track))

        ranked.sort(key=lambda item: item[0], reverse=True)
        logger.info("Ranked %d candidate tracks for score=%.2f (%s)", len(ranked), score, state_label)
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
            self._reccobeats_call_count += 1
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
            self._reccobeats_call_count += 1
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
        ema_score: float,
        track: TrackSummary,
        features: dict[str, float] | None,
        profile: TasteProfile,
    ) -> float | None:
        if features is None:
            logger.debug("Skipping track %s because audio features were unavailable", track.track_id)
            return None

        target = self._target_audio_profile(ema_score)
        energy_score = 1.0 - abs(features["energy"] - target["energy"])
        valence_score = 1.0 - abs(features["valence"] - target["valence"])
        tempo_delta = abs(features["tempo"] - target["tempo"]) / 80.0
        tempo_score = max(0.0, 1.0 - tempo_delta)

        artist_match = 0.15 if any(name in track.artist for name in profile.seed_artist_names[:5]) else 0.0
        freshness_bonus = 0.1 if track.track_id not in profile.excluded_track_ids else -0.5
        recommendation_penalty = -0.6 if track.track_id in self._recent_recommendation_ids else 0.0
        artist_repeat_penalty = self._artist_repeat_penalty(track.artist)

        return (
            (energy_score * 0.4)
            + (valence_score * 0.3)
            + (tempo_score * 0.3)
            + artist_match
            + freshness_bonus
            + recommendation_penalty
            + artist_repeat_penalty
        )

    def _genres_for_score(self, ema_score: float) -> list[str]:
        score = clamp_score(ema_score)
        if score < DROWSY_SCORE_THRESHOLD:
            return AWAKE_GENRES
        if score < TIRED_SCORE_THRESHOLD:
            return DROWSY_GENRES
        return TIRED_GENRES

    def _target_audio_profile(self, ema_score: float) -> dict[str, float]:
        score = clamp_score(ema_score)
        return {
            "energy": 0.55 + (0.35 * score),
            "valence": 0.55 + (0.20 * score),
            "tempo": 108.0 + (22.0 * score),
        }

    def _track_from_spotify(self, item: dict[str, Any]) -> TrackSummary:
        artists = item.get("artists") or [{"name": "Unknown Artist"}]
        return TrackSummary(
            track_id=item.get("id", "unknown-track"),
            name=item.get("name", "Unknown Track"),
            artist=", ".join(artist.get("name", "Unknown Artist") for artist in artists),
            uri=item.get("uri", ""),
        )

    def _remember_recommendations(self, tracks: list[TrackSummary]) -> None:
        for track in tracks:
            self._recent_recommendation_ids.append(track.track_id)
            for artist_name in track.artist.split(", "):
                normalized = artist_name.strip()
                if normalized:
                    self._recent_recommendation_artists.append(normalized)

    def _artist_repeat_penalty(self, artist_text: str) -> float:
        penalty = 0.0
        recent_artists = list(self._recent_recommendation_artists)
        for artist_name in artist_text.split(", "):
            normalized = artist_name.strip()
            if not normalized:
                continue
            occurrences = recent_artists.count(normalized)
            penalty -= min(0.12 * occurrences, 0.36)
        return penalty

    def _query_offset(self, query_index: int) -> int:
        cycle = self._refresh_generation % SEARCH_OFFSET_CYCLE
        return (cycle * SEARCH_OFFSET_STEP) + ((query_index % 2) * SEARCH_OFFSET_STEP)

    def _set_snapshot(self, snapshot: SpotifySnapshot) -> None:
        with self._state_lock:
            self.snapshot = snapshot

    def _fallback_tracks(self, ema_score: float = 0.2, limit: int = 4) -> list[TrackSummary]:
        state_label = fatigue_state_from_score(ema_score)
        defaults = {
            "awake": [
                ("fallback-awake-1", "Deep Focus Loop", "Study Buddy"),
                ("fallback-awake-2", "Quiet Momentum", "Study Buddy"),
                ("fallback-awake-3", "Steady Pixels", "Study Buddy"),
                ("fallback-awake-4", "Flow State", "Study Buddy"),
            ],
            "drowsy": [
                ("fallback-drowsy-1", "Fresh Tab", "Study Buddy"),
                ("fallback-drowsy-2", "Level Up", "Study Buddy"),
                ("fallback-drowsy-3", "Color Burst", "Study Buddy"),
                ("fallback-drowsy-4", "New Loop", "Study Buddy"),
            ],
            "tired": [
                ("fallback-tired-1", "Wake Up Sprint", "Study Buddy"),
                ("fallback-tired-2", "Bright Notes", "Study Buddy"),
                ("fallback-tired-3", "Second Wind", "Study Buddy"),
                ("fallback-tired-4", "Keep Moving", "Study Buddy"),
            ],
        }
        return [
            TrackSummary(track_id=track_id, name=name, artist=artist, uri=f"spotify:track:{track_id}")
            for track_id, name, artist in defaults[state_label][:limit]
        ]
