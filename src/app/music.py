import requests


import spotipy
from spotipy.oauth2 import SpotifyOAuth

from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

REDIRECT_URI = 'http://127.0.0.1:8888/callback'

# 2. Define Scope (modify-playback allows adding to queue)
scope = "user-modify-playback-state user-read-private"

# 3. Authenticate using SpotifyOAuth
# This will open a browser window the first time you run it
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=scope,
    cache_path=".cache" # Stores your token so you don't login every time
))

def add_to_queue(track_id):
    try:
        # track_id can be a URI: 'spotify:track:ID'
        sp.add_to_queue(uri=track_id)
        print("Successfully queued!")
    except Exception as e:
        print(f"Error: {e}")

# Test it
add_to_queue('spotify:track:5Qv2Nby1xTr9pQyjkrc94J')