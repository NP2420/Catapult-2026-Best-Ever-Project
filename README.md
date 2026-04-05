# AI Study Buddy

A desktop robot study buddy that monitors your focus and dynamically builds a Spotify queue based on it.

Featuring:

- webcam focus/drowsiness state polling via a finetuned YOLO model
- SpotifyAPI integration and a custom-build recommendation algorithm
- realtime inference and Spotify queue appending
- an always-on-top pixel buddy UI (its name is john/jane studybuddy)

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

Offline/demo mode without Spotify OAuth or API traffic:

```bash
python src/app/music.py --no-spotify
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

Without those values, the app still runs and falls back to local demo tracks.
You can also force that behavior with `--no-spotify`, which is useful for users without Spotify and for debugging the model/break flow without hitting the Spotify API.

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

# ABOUT

## Inspiration

We really wanted to make something that was fun, unique, and something we'd want to use ourselves. A desktop buddy that makes studying more interesting and involves computer vision, machine learning, and music was a strong fit for these goals. More specifically, the idea of adaptive song selection based on the user's current emotions/state was something we wanted from the very beginning.

## What it does

It's a cute little robot buddy that sits on your screen, and you can drag it around wherever you want. It regularly pulls webcam data through a fine-tuned YOLO model to detect drowsiness/focus level on a scale of 0-1 through facial features, and then periodically tells a song ranking algorithm to find a song that matches the current level (through audio features like energy, valence, and tempo) and add it to the queue.

## How we built it

Daniel was primarily responsible for the app functionality and Spotify API integration, while Nathan was responsible for the model training, evaluation, and real-time inference. We used a pyside Qt frontend in order to get the desktop buddy window, open CV for webcam capture, fine tuning a YOLO model on a drowsiness detection dataset on the RCAC gautschi clusters, and SpotifyAPI for realtime song selection and queuing based on the user’s drowsiness.

## Challenges we ran into

Both Nathan and I ran into significant challenges. The most significant one for me was realizing less than 15 hours before the submission deadline that the spotify API endpoint for recommendations was deprecated. This led to me needing to design an entirely custom recommendation system using the limited SpotifyAPI endpoints available, and though it isn’t as polished as Spotify's algorithm, it worked enough to develop this MVP.

The initial dataset Nathan chose, as well as the model architecture, did not yield great results in the beginning. This may have been due to it being multiclass classification, as well as training from scratch. He made the decision to downscope the ML side, opting for a fine-tuning approach and a simpler output value, which ended up yielding great results, as our model performs very well at detecting alertness and drowsiness, even when running purely on local machines.

## Accomplishments that we're proud of

We were really proud of actually getting a final product out, actually meeting the goals we set at the beginning of the hackathon, and overcoming the seriously difficult challenges we faced throughout this weekend.

## What we learned

Start early, do your research on the tools you are going to use, and don’t be afraid to pivot and downscope a bit in order to deliver a final product.

## What's next for AI Study Buddy

There are many additional features that can be added to AI Study Buddy. A line graph visualization to show focus value over time would be very interesting and cool to see as a user. Also, working on improving that song recommendation algorithm, or swapping APIs entirely to a music service that is more open for its developers. Last but not least, we had a model trained to detect emotions as well, so it is very possible to integrate emotion predictions into our song recommendation parameters to get even more personalized music recommendations.
