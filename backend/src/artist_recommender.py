import os
import sys

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.song_lookup import normalize_text


class ArtistSimilarityRecommender:
    """
    Artist Similarity Recommender module for Spotify Recommendation System.

    IMPORTANT ARCHITECTURAL NOTE:
    This recommender uses TF-IDF vectorization on artist name strings.
    It measures text/token overlap among artist credits and collaborations
    (e.g., identifying tracks that share collaborating artists).
    It does NOT measure acoustic or musical similarity between artist catalogs.
    Artist musical similarity based on aggregated audio profiles is recommended
    for a future phase.
    """
    def __init__(self, cleaned_data_path=None, model_dir="models"):
        if cleaned_data_path is None:
            if os.path.exists("data/cleaned_spotify.csv"):
                cleaned_data_path = "data/cleaned_spotify.csv"
            else:
                cleaned_data_path = "data/processed/train.csv"

        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "artist_similarity.pkl")
        self.vectorizer_path = os.path.join(model_dir, "artist_vectorizer.pkl")

        # Placeholders for lazy loading
        self.unique_artists = None       # Pandas Series of unique artist strings
        self.vectorizer = None           # Fitted TfidfVectorizer
        self.tfidf_matrix = None         # Sparse TF-IDF feature matrix of unique artists
        self.artist_to_idx = None        # Map from normalized artist name to matrix row index

    def load_data_and_models(self):
        """
        Lazy-loads unique artists and the TF-IDF representation.
        If precomputed state exists in models/artist_similarity.pkl, loads it.
        Otherwise, fits TF-IDF vectorizer on unique artist names and caches it.
        """
        os.makedirs(self.model_dir, exist_ok=True)

        # 1. Attempt to load cached state
        if os.path.exists(self.model_path):
            print(f"[Lazy Load] Loading precomputed artist similarity state from {self.model_path}...")
            try:
                with open(self.model_path, "rb") as f:
                    saved_model = pickle.load(f)
                    self.unique_artists = saved_model["unique_artists"]
                    self.vectorizer = saved_model["vectorizer"]
                    self.tfidf_matrix = saved_model["tfidf_matrix"]
                    self.artist_to_idx = saved_model["artist_to_idx"]
                print("[Lazy Load] Artist similarity model loaded successfully.")
                return
            except Exception as e:
                print(f"[Warning] Failed to load cached model: {e}. Rebuilding...")

        # 2. Lazy load artists column from dataset
        print(f"[Lazy Load] Loading artist column from {self.cleaned_data_path}...")
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.cleaned_data_path}. Please run preprocess.py first."
            )

        df = pd.read_csv(self.cleaned_data_path, usecols=["artists"])
        df = df.dropna(subset=["artists"])
        df["artists"] = df["artists"].str.strip()

        # 3. Extract unique artist strings sorted for determinism
        unique_list = sorted(df["artists"].unique())
        self.unique_artists = pd.Series(unique_list, name="artists")

        # 4. Apply TF-IDF Vectorizer
        print("Applying TF-IDF Vectorizer to unique artist names...")
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.unique_artists)

        # 5. Build lookup dictionary with normalized keys for fast deterministic resolution
        self.artist_to_idx = {
            normalize_text(name): idx for idx, name in enumerate(self.unique_artists)
        }

        # 6. Save model state
        self.save_model()

    def save_model(self):
        """
        Saves the fitted model state and standalone vectorizer to disk using pickle.
        """
        model_state = {
            "unique_artists": self.unique_artists,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "artist_to_idx": self.artist_to_idx,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(model_state, f)

        # Also save standalone vectorizer for interoperability
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Artist similarity model saved to {self.model_path} ({len(self.unique_artists)} unique artists)")

    def get_similar_artists(self, artist_name, top_n=10):
        """
        Returns the top N most similar artists to the query artist_name based on TF-IDF name overlap.
        Computes cosine similarity on-the-fly for rapid query execution and low memory usage.
        Deterministic tie-breaking is enforced.
        """
        if self.tfidf_matrix is None:
            self.load_data_and_models()

        query_clean = normalize_text(artist_name)
        if not query_clean:
            return None, "Artist name cannot be blank."

        # 1. Exact match lookup
        if query_clean in self.artist_to_idx:
            artist_idx = self.artist_to_idx[query_clean]
        else:
            # Fallback to deterministic partial match
            # Rank candidates: exact word in artist string, shorter name length, alphabetical
            matches = [
                name for name in self.artist_to_idx
                if query_clean in name
            ]
            if not matches:
                return None, f"No artist matching '{artist_name}' found in the dataset."

            matches = sorted(
                matches,
                key=lambda m: (
                    0 if f" {query_clean} " in f" {m} " else 1,
                    len(m),
                    m
                )
            )
            matched_artist = matches[0]
            artist_idx = self.artist_to_idx[matched_artist]
            print(f"[Search Match] Exact match for '{artist_name}' not found. Using closest match: '{self.unique_artists.iloc[artist_idx]}'")

        # 2. Extract query artist's TF-IDF vector
        query_vector = self.tfidf_matrix[artist_idx]

        # 3. Calculate Cosine Similarity on-the-fly
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # 4. Sort indices in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # 5. Retrieve top N recommendations
        recommendations = []
        for idx in sorted_indices:
            if idx == artist_idx:
                continue

            score = float(similarity_scores[idx])
            if score <= 0.0:
                continue

            rec_artist = self.unique_artists.iloc[idx]
            recommendations.append({
                "artist": rec_artist,
                "similarity_score": score
            })

            if len(recommendations) == top_n:
                break

        return pd.DataFrame(recommendations), None


# ==============================================================================
# Package-level Wrapper Function
# ==============================================================================

_global_artist_recommender = None


def get_similar_artists(artist_name, top_n=10):
    """
    Module level helper function matching the get_similar_artists requirement.
    Lazily initializes the similarity recommender instance.
    """
    global _global_artist_recommender
    if _global_artist_recommender is None:
        _global_artist_recommender = ArtistSimilarityRecommender()

    return _global_artist_recommender.get_similar_artists(artist_name, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("           ARTIST SIMILARITY TESTING BLOCK            ")
    print("=" * 60)

    test_artist = "Lata Mangeshkar"
    print(f"Querying similar artists to: '{test_artist}'...\n")

    try:
        rec_df, err = get_similar_artists(test_artist, top_n=10)
        if err:
            print(f"Error: {err}")
        else:
            print(f"Top 10 similar artists to '{test_artist}':")
            if rec_df.empty:
                print("No overlapping artist names (collaborations) found.")
            else:
                print(rec_df.to_string(index=False))
    except Exception as e:
        print(f"Failed to run recommendation engine: {e}")
    print("=" * 60)
