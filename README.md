# AI Study Buddy

Desktop study companion MVP with:

- webcam mood polling
- smoothed rule-based productivity logic
- Spotify recommendation and queue refresh hooks
- an always-on-top pixel buddy UI

## Run

1. Setup Dependencies

```bash
uv sync
source .venv/scripts/activate # bash command, may be different in powershell or other kinds of shells
```

2. Run Application

```bash
python src/app/music.py
```

## Environment

Set Spotify credentials if you want live playback integration (requires premium unfortunately):

1. Go to Spotify Developer Dashboard
2. Create an app
3. Add http://localhost:8888/callback as redirect URI
4. Copy your Client ID and Client Secret into .env
5. Run the app

```bash
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

Optional tuning and logging variables:

```bash
STUDY_BUDDY_LOG_LEVEL=INFO
STUDY_BUDDY_CAMERA_INDEX=0
STUDY_BUDDY_MOOD_POLL_SECONDS=1.0
STUDY_BUDDY_MOOD_SMOOTHING_WINDOW=10
STUDY_BUDDY_TICK_MS=1000
STUDY_BUDDY_QUEUE_REFRESH_SECONDS=45
STUDY_BUDDY_SPOTIFY_LIMIT=4
STUDY_BUDDY_SHORT_TERM_THRESHOLD=15
STUDY_BUDDY_LONG_TERM_THRESHOLD=90
STUDY_BUDDY_MIN_BREAK_SECONDS=300
STUDY_BUDDY_FRAME_INTERVAL_MS=300
STUDY_BUDDY_BUDDY_SIZE_PX=96
STUDY_BUDDY_WINDOW_MARGIN_PX=24
```

Without those values, the app still runs and falls back to local demo tracks.
