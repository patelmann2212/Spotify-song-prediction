"""
Automated tests for the SoundScout Spotify Recommendation API.
Run with:  pytest tests/test_api.py -v
"""
import pytest
import math
from fastapi.testclient import TestClient
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api import app


@pytest.fixture(scope="module")
def client():
    """Single TestClient that loads engines once for all tests."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(["content", "genre", "artist", "mood", "weighted"]).issubset(set(body["engines"]))


# ---------------------------------------------------------------------------
# /recommend — validation
# ---------------------------------------------------------------------------

def test_invalid_engine(client):
    r = client.post("/recommend", json={"engine": "psychic", "song": "Hello"})
    assert r.status_code == 400
    assert "Invalid engine" in r.json()["detail"]

def test_content_missing_song(client):
    r = client.post("/recommend", json={"engine": "content", "song": ""})
    assert r.status_code == 400
    assert "Song name is required" in r.json()["detail"]

def test_mood_missing_mood(client):
    r = client.post("/recommend", json={"engine": "mood", "mood": ""})
    assert r.status_code == 400

def test_artist_missing_artist(client):
    r = client.post("/recommend", json={"engine": "artist", "artist": ""})
    assert r.status_code == 400

def test_unknown_song(client):
    r = client.post("/recommend", json={"engine": "content", "song": "XXXNONEXISTENTSONGXXX"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /recommend — engines return real data
# ---------------------------------------------------------------------------

def test_mood_happy(client):
    r = client.post("/recommend", json={"engine": "mood", "mood": "Happy", "top_n": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) >= 1
    for item in results:
        assert "track_name" in item
        assert "popularity" in item

def test_mood_party(client):
    r = client.post("/recommend", json={"engine": "mood", "mood": "Party", "top_n": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1

def test_mood_invalid(client):
    r = client.post("/recommend", json={"engine": "mood", "mood": "Angry"})
    assert r.status_code == 404
    assert "Invalid mood" in r.json()["detail"]

def test_content_blinding_lights(client):
    r = client.post("/recommend", json={"engine": "content", "song": "Blinding Lights", "top_n": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) >= 1
    for item in results:
        assert "track_name" in item
        assert "artists" in item
        assert "similarity_score" in item
        score = item["similarity_score"]
        assert score is not None
        assert 0.0 <= score <= 1.0
        assert not math.isnan(score)
        # Input song must not appear in results
        assert item["track_name"] != "Blinding Lights"

def test_content_case_insensitive(client):
    """Song lookup should be case-insensitive."""
    r1 = client.post("/recommend", json={"engine": "content", "song": "Blinding Lights", "top_n": 3})
    r2 = client.post("/recommend", json={"engine": "content", "song": "blinding lights", "top_n": 3})
    assert r1.status_code == 200
    assert r2.status_code == 200
    titles1 = [x["track_name"] for x in r1.json()["results"]]
    titles2 = [x["track_name"] for x in r2.json()["results"]]
    assert titles1 == titles2

def test_genre_returns_fields(client):
    r = client.post("/recommend", json={"engine": "genre", "song": "Blinding Lights", "top_n": 5})
    assert r.status_code in (200, 404)  # 404 if not in dataset
    if r.status_code == 200:
        for item in r.json()["results"]:
            assert "track_genre" in item

def test_artist_engine(client):
    r = client.post("/recommend", json={"engine": "artist", "artist": "The Weeknd", "top_n": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) >= 1
    for item in results:
        assert "similarity_score" in item
        score = item["similarity_score"]
        assert 0.0 <= score <= 1.0

def test_weighted_engine(client):
    r = client.post("/recommend", json={"engine": "weighted", "song": "Blinding Lights", "top_n": 8})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) >= 1
    for item in results:
        assert "final_score" in item
        score = item["final_score"]
        assert score is not None
        assert not math.isnan(score)

def test_no_duplicate_results(client):
    r = client.post("/recommend", json={"engine": "content", "song": "Blinding Lights", "top_n": 8})
    assert r.status_code == 200
    names = [x["track_name"] for x in r.json()["results"]]
    assert len(names) == len(set(names)), "Duplicate track names in results"

def test_scores_are_finite(client):
    for engine, payload in [
        ("content",  {"engine": "content",  "song": "Blinding Lights"}),
        ("weighted", {"engine": "weighted", "song": "Blinding Lights"}),
    ]:
        r = client.post("/recommend", json={**payload, "top_n": 5})
        if r.status_code != 200:
            continue
        for item in r.json()["results"]:
            for field in ["similarity_score", "final_score"]:
                if item.get(field) is not None:
                    assert math.isfinite(item[field]), f"{field} is not finite: {item[field]}"


# ---------------------------------------------------------------------------
# /search autocomplete
# ---------------------------------------------------------------------------

def test_search_endpoint(client):
    r = client.get("/search?q=blind&limit=5")
    assert r.status_code == 200
    results = r.json()["results"]
    assert isinstance(results, list)
    assert len(results) <= 5

def test_search_empty_query(client):
    r = client.get("/search?q=")
    assert r.status_code == 400
