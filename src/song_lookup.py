"""
Song and Artist Lookup Utility for Spotify Recommendation System.

Provides deterministic, case-insensitive, and punctuation-tolerant
matching for songs and artists, eliminating arbitrary first-row selections.
"""

import re
import pandas as pd
from typing import Optional, Tuple


def normalize_text(text: Optional[str]) -> str:
    """
    Safely normalizes strings for text matching and comparison:
    - Casts to lowercase
    - Strips leading and trailing whitespace
    - Condenses multiple consecutive whitespace characters to a single space
    - Strips common punctuation characters while retaining alphanumeric characters

    Original display values should never be mutated by this function.
    """
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove punctuation characters (preserving alphanumeric and whitespace)
    text = re.sub(r"[^\w\s]", "", text)
    # Condense consecutive whitespace to a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_base_title(title: Optional[str]) -> str:
    """
    Extracts base title by removing common parenthetical/bracketed version tags
    like (Remix), [Live], - Remastered, etc., and normalizing.
    Useful for group-aware track splitting so versions don't leak across splits.
    """
    if title is None or not isinstance(title, str):
        return ""
    # Remove contents inside (), [], or {}
    t = re.sub(r"[\(\[\{].*?[\)\]\}]", "", title)
    # Remove trailing tags after hyphen like " - Remastered 2021" or " - Live"
    t = re.sub(r"\s+-\s+.*$", "", t)
    return normalize_text(t)


def find_song(
    df: pd.DataFrame,
    song_name: str,
    artist_name: Optional[str] = None
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Deterministically resolves a query song (and optional artist) to a single track row.

    Ranking Strategy:
    1. Exact normalized title match takes precedence over partial contains match.
    2. When artist is provided:
       - Exact artist match (score 2)
       - Sub-artist / token match in collaboration (score 1)
       - Non-matching artists are excluded so an unrelated artist is never silently picked.
    3. Popularity descending (e.g. The Weeknd pop 91 vs Kidz Bop pop 0).
    4. Stable deterministic tie-breaker: track_id ascending (never relies on dataframe row order).

    Returns:
        (best_match_series, error_message)
    """
    if not song_name or not isinstance(song_name, str) or not song_name.strip():
        return None, "Song name cannot be empty."

    norm_query_song = normalize_text(song_name)
    if not norm_query_song:
        return None, "Song name cannot be empty."

    norm_query_artist = normalize_text(artist_name) if artist_name and artist_name.strip() else None

    # Compute normalized columns for search
    norm_titles = df["track_name"].fillna("").apply(normalize_text)
    norm_artists = df["artists"].fillna("").apply(normalize_text)

    # 1. Title matching: exact title first, then partial contains fallback
    exact_title_mask = norm_titles == norm_query_song
    contains_title_mask = norm_titles.str.contains(re.escape(norm_query_song), regex=True)

    if exact_title_mask.any():
        matched_indices = df[exact_title_mask].index
    elif contains_title_mask.any():
        matched_indices = df[contains_title_mask].index
    else:
        return None, f"No song matching '{song_name}' found in the dataset."

    candidates = df.loc[matched_indices].copy()
    candidates["_norm_title"] = norm_titles.loc[matched_indices]
    candidates["_norm_artist"] = norm_artists.loc[matched_indices]
    candidates["_exact_title"] = (candidates["_norm_title"] == norm_query_song).astype(int)

    # Ensure popularity column exists for ranking
    if "popularity" not in candidates.columns:
        candidates["popularity"] = 0

    # Ensure track_id exists for deterministic tie-breaking
    if "track_id" not in candidates.columns:
        candidates["track_id"] = candidates.index.astype(str)

    if norm_query_artist:
        def calc_artist_score(candidate_norm_artist: str) -> int:
            if candidate_norm_artist == norm_query_artist:
                return 2
            # Handle multi-artist strings separated by semicolons or commas
            sub_artists = [normalize_text(p) for p in re.split(r"[;,]", candidate_norm_artist)]
            if norm_query_artist in sub_artists:
                return 2
            # Check if query artist is contained within the string or vice-versa
            if norm_query_artist in candidate_norm_artist or candidate_norm_artist in norm_query_artist:
                return 1
            return 0

        candidates["_artist_score"] = candidates["_norm_artist"].apply(calc_artist_score)

        # Filter out candidates with zero artist match
        artist_matches = candidates[candidates["_artist_score"] > 0]
        if artist_matches.empty:
            return None, f"No song matching '{song_name}' by '{artist_name}' found in the dataset."
        candidates = artist_matches

        # Rank: artist score desc -> exact title desc -> popularity desc -> track_id asc
        candidates = candidates.sort_values(
            by=["_artist_score", "_exact_title", "popularity", "track_id"],
            ascending=[False, False, False, True]
        )
    else:
        # Title only: exact title desc -> popularity desc -> track_id asc
        candidates = candidates.sort_values(
            by=["_exact_title", "popularity", "track_id"],
            ascending=[False, False, True]
        )

    best_match = candidates.iloc[0]
    return best_match, None


def get_song_candidates(
    df: pd.DataFrame,
    song_name: str,
    artist_name: Optional[str] = None,
    top_n: int = 5
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Returns up to top_n candidate matches sorted by deterministic relevance.
    """
    if not song_name or not isinstance(song_name, str) or not song_name.strip():
        return None, "Song name cannot be empty."

    norm_query_song = normalize_text(song_name)
    if not norm_query_song:
        return None, "Song name cannot be empty."

    norm_query_artist = normalize_text(artist_name) if artist_name and artist_name.strip() else None

    norm_titles = df["track_name"].fillna("").apply(normalize_text)
    norm_artists = df["artists"].fillna("").apply(normalize_text)

    exact_title_mask = norm_titles == norm_query_song
    contains_title_mask = norm_titles.str.contains(re.escape(norm_query_song), regex=True)

    if exact_title_mask.any():
        matched_indices = df[exact_title_mask].index
    elif contains_title_mask.any():
        matched_indices = df[contains_title_mask].index
    else:
        return None, f"No song matching '{song_name}' found in the dataset."

    candidates = df.loc[matched_indices].copy()
    candidates["_norm_title"] = norm_titles.loc[matched_indices]
    candidates["_norm_artist"] = norm_artists.loc[matched_indices]
    candidates["_exact_title"] = (candidates["_norm_title"] == norm_query_song).astype(int)

    if "popularity" not in candidates.columns:
        candidates["popularity"] = 0
    if "track_id" not in candidates.columns:
        candidates["track_id"] = candidates.index.astype(str)

    if norm_query_artist:
        def calc_artist_score(candidate_norm_artist: str) -> int:
            if candidate_norm_artist == norm_query_artist:
                return 2
            sub_artists = [normalize_text(p) for p in re.split(r"[;,]", candidate_norm_artist)]
            if norm_query_artist in sub_artists:
                return 2
            if norm_query_artist in candidate_norm_artist or candidate_norm_artist in norm_query_artist:
                return 1
            return 0

        candidates["_artist_score"] = candidates["_norm_artist"].apply(calc_artist_score)
        artist_matches = candidates[candidates["_artist_score"] > 0]
        if artist_matches.empty:
            return None, f"No song matching '{song_name}' by '{artist_name}' found in the dataset."
        candidates = artist_matches
        candidates = candidates.sort_values(
            by=["_artist_score", "_exact_title", "popularity", "track_id"],
            ascending=[False, False, False, True]
        )
    else:
        candidates = candidates.sort_values(
            by=["_exact_title", "popularity", "track_id"],
            ascending=[False, False, True]
        )

    cols_to_return = [c for c in ["track_id", "track_name", "artists", "album_name", "popularity", "track_genre"] if c in candidates.columns]
    return candidates[cols_to_return].head(top_n).reset_index(drop=True), None
