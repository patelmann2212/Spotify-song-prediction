from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import math
import json
import os
import sys
import numpy as np

# Ensure root of project is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mood_recommender import MoodRecommender
from src.weighted_recommender import WeightedRecommender
from src.song_lookup import get_song_candidates

# ---------------------------------------------------------------------------
# Engine registry (populated at startup)
# ---------------------------------------------------------------------------
engines: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all engines once at startup; clean up on shutdown."""
    print("[API] Initializing recommendation engines...")

    # WeightedRecommender internally initializes ContentBased, Genre, and Artist engines
    weighted = WeightedRecommender()
    weighted.load_engines()

    mood = MoodRecommender()
    mood.load_data_and_classify()

    engines["weighted"] = weighted
    engines["content"] = weighted.audio_engine
    engines["genre"] = weighted.genre_engine
    engines["artist"] = weighted.artist_engine
    engines["mood"] = mood

    print("[API] All engines ready.")
    yield
    engines.clear()


app = FastAPI(title="Spotify Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _make_serializable(value):
    """Convert numpy / pandas scalars and non-finite floats to JSON-safe types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    return value


def clean_df(df, keep_cols=None):
    """Convert a DataFrame to a JSON-safe list of dicts.

    Args:
        df: pandas DataFrame
        keep_cols: optional list of columns to include; None means all
    """
    if keep_cols:
        df = df[[c for c in keep_cols if c in df.columns]]
    records = df.fillna("").to_dict(orient="records")
    return [
        {k: _make_serializable(v) for k, v in row.items()}
        for row in records
    ]


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    engine: str
    song: str = ""
    artist: str = ""
    mood: str = ""
    top_n: int = 8


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "engines": list(engines.keys())}


@app.get("/search")
def search_songs(q: str, limit: int = 6):
    """Return up to `limit` song title/artist matches for autocomplete."""
    if not engines:
        raise HTTPException(status_code=503, detail="Engines not yet loaded.")
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    weighted: WeightedRecommender = engines.get("weighted")
    if weighted is None or weighted.df is None:
        raise HTTPException(status_code=503, detail="Dataset unavailable.")

    candidates, err = get_song_candidates(weighted.df, q, top_n=limit)
    if err:
        return {"results": []}

    return {"results": clean_df(candidates, keep_cols=["track_id", "track_name", "artists", "track_genre", "popularity"])}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    engine_name = req.engine.lower()

    if not engines:
        raise HTTPException(status_code=503, detail="Recommendation engines are not yet loaded.")

    if engine_name not in engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine '{engine_name}'. Valid: content, genre, artist, mood, weighted.")

    top_n = max(1, min(req.top_n, 20))

    try:
        if engine_name == "artist":
            if not req.artist or not req.artist.strip():
                raise HTTPException(status_code=400, detail="Artist name is required for the Artist engine.")
            df, err = engines["artist"].get_similar_artists(req.artist.strip(), top_n=top_n)

        elif engine_name == "mood":
            if not req.mood or not req.mood.strip():
                raise HTTPException(status_code=400, detail="Mood is required. Valid moods: Happy, Sad, Party, Chill.")
            df, err = engines["mood"].recommend_by_mood(req.mood.strip(), top_n=top_n)

        else:  # content, genre, weighted
            if not req.song or not req.song.strip():
                raise HTTPException(status_code=400, detail="Song name is required.")
            artist_filter = req.artist.strip() if req.artist and req.artist.strip() else None

            if engine_name == "content":
                df, err = engines["content"].get_song_recommendations(req.song.strip(), artist_filter, top_n=top_n)
            elif engine_name == "genre":
                df, err = engines["genre"].recommend_by_genre(req.song.strip(), artist_filter, top_n=top_n)
            elif engine_name == "weighted":
                df, err = engines["weighted"].get_weighted_recommendations(req.song.strip(), artist_filter, top_n=top_n)

        if err:
            raise HTTPException(status_code=404, detail=err)

        if df is None or df.empty:
            return {"results": [], "message": "No recommendations found for this query."}

        # For mood engine, only expose clean metadata columns
        mood_cols = ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]
        general_cols = ["track_name", "artists", "album_name", "track_genre", "popularity",
                        "similarity_score", "final_score", "audio_similarity", "artist_similarity",
                        "genre_similarity", "popularity_score"]
        artist_cols = ["artist", "similarity_score"]

        if engine_name == "mood":
            result_cols = mood_cols
        elif engine_name == "artist":
            result_cols = artist_cols
        else:
            result_cols = general_cols

        return {"results": clean_df(df, keep_cols=result_cols)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
