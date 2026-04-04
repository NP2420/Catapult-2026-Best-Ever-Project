# AI Study Buddy

Desktop study companion MVP with:

- webcam mood polling
- smoothed rule-based productivity logic
- Spotify recommendation and queue refresh hooks
- an always-on-top pixel buddy UI

## Run

```bash
python src/app/music.py
```

## Environment

Set Spotify credentials if you want live playback integration:

```bash
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

Without those values, the app still runs and falls back to local demo tracks.
