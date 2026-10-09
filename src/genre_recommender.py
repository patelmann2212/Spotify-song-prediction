import os
import sys

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from src.song_lookup import find_song, normalize_text


class GenreRecommender:
    """
    Genre Recommendation Engine for the Spotify Song Prediction system.

    This engine leverages the 'track_genre' column of the dataset to:
    1. Encode genres:
       - Categorical One-Hot Encoding for song-level genre representation.
       - Audio Profile Encoding: Computes a mean feature vector for each genre
         based on the audio features of songs belonging to it.
    2. Calculate genre similarity:
       - Computes pairwise cosine similarity between all genres using their mean audio profiles.
    3. Recommend songs:
       - Restricts recommendations to the same genre as the query song.
       - Ranks same-genre candidate songs by their audio similarity to the query song deterministically.
    """
    def __init__(self, cleaned_data_path=None, model_dir="models"):
        if cleaned_data_path is None:
            # Default to full catalog if available, otherwise train set
            if os.path.exists("data/cleaned_spotify.csv"):
                cleaned_data_path = "data/cleaned_spotify.csv"
            else:
                cleaned_data_path = "data/processed/train.csv"

        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "genre_recommender.pkl")

        # Placeholders for lazy loading
        self.df = None
        self.scaled_feature_cols = None
        self.unique_genres = None
        self.genre_to_idx = None
        self.genre_profiles = None
        self.genre_similarity_matrix = None
        self.one_hot_encoder = None

    def load_data_and_models(self):
        """
        Loads the preprocessed dataset and initializes or loads precomputed genre matrices.
        If a saved model state is found in models/genre_recommender.pkl, loads it.
        Otherwise, builds the profiles and similarity matrices on-the-fly and caches them.
        """
        os.makedirs(self.model_dir, exist_ok=True)

        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.cleaned_data_path}. Please run preprocess.py first."
            )

        print(f"[GenreRecommender] Loading dataset from {self.cleaned_data_path}...")
        self.df = pd.read_csv(self.cleaned_data_path).reset_index(drop=True)

        # Identify features starting with 'scaled_' for similarity calculations
        self.scaled_feature_cols = [
            col for col in self.df.columns if col.startswith("scaled_")
        ]

        # Check if cache exists
        if os.path.exists(self.model_path):
            print(f"[GenreRecommender] Loading cached models from {self.model_path}...")
            try:
                with open(self.model_path, "rb") as f:
                    cache = pickle.load(f)
                    self.unique_genres = cache["unique_genres"]
                    self.genre_to_idx = cache["genre_to_idx"]
                    self.genre_profiles = cache["genre_profiles"]
                    self.genre_similarity_matrix = cache["genre_similarity_matrix"]
                    self.one_hot_encoder = cache["one_hot_encoder"]
                print("[GenreRecommender] Loaded cached state successfully.")
                return
            except Exception as e:
                print(f"[Warning] Failed to load cached model state: {e}. Rebuilding...")

        # Build model if not cached
        print("[GenreRecommender] Encoding genres and calculating similarities...")
        self.encode_genres()
        self.calculate_genre_similarity()
        self.save_model()

    def encode_genres(self):
        """
        Performs two encodings of the track genres:
        1. Song-Level One-Hot Encoding: Creates binary vectors for genres using scikit-learn.
        2. Genre-Level Audio Profiling: Computes a mean audio vector for each genre.
        """
        if "track_genre" not in self.df.columns:
            raise KeyError("The 'track_genre' column is missing from the dataset.")

        self.df["track_genre"] = self.df["track_genre"].astype(str).str.strip()

        # 1. Categorical One-Hot Encoding
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.one_hot_encoder.fit(self.df[["track_genre"]])

        # 2. Genre-Level Audio Profiling
        genre_profiles_df = self.df.groupby("track_genre")[self.scaled_feature_cols].mean()

        self.unique_genres = sorted(genre_profiles_df.index.tolist())
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.unique_genres)}

        # Ensure profiles matrix aligns with sorted unique_genres
        self.genre_profiles = genre_profiles_df.loc[self.unique_genres].values
        print(f"  -> Encoded {len(self.unique_genres)} unique genres.")

    def calculate_genre_similarity(self):
        """
        Calculates a genre-to-genre similarity matrix.
        Uses Cosine Similarity of the mean audio feature profiles for each genre.
        """
        print("  -> Calculating pairwise genre similarity matrix...")
        self.genre_similarity_matrix = cosine_similarity(self.genre_profiles)
        print("  -> Similarity matrix calculated successfully.")

    def save_model(self):
        """
        Saves the fitted encoders, unique genres list, profiles, and similarity matrix to disk.
        """
        cache = {
            "unique_genres": self.unique_genres,
            "genre_to_idx": self.genre_to_idx,
            "genre_profiles": self.genre_profiles,
            "genre_similarity_matrix": self.genre_similarity_matrix,
            "one_hot_encoder": self.one_hot_encoder,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[GenreRecommender] Model saved to {self.model_path}")

    def get_similar_genres(self, genre_name, top_n=5):
        """
        Retrieves the top_n most similar genres to a given genre using the similarity matrix.
        """
        if self.genre_similarity_matrix is None:
            self.load_data_and_models()

        query_genre = normalize_text(genre_name)
        matched = [g for g in self.unique_genres if normalize_text(g) == query_genre]
        if not matched:
            return None, f"Genre '{genre_name}' not found in the dataset."

        matched_genre = matched[0]
        genre_idx = self.genre_to_idx[matched_genre]

        similarities = self.genre_similarity_matrix[genre_idx]
        sorted_indices = np.argsort(similarities)[::-1]

        similar_genres = []
        for idx in sorted_indices:
            if idx == genre_idx:
                continue
            similar_genres.append({
                "genre": self.unique_genres[idx],
                "similarity_score": float(similarities[idx])
            })
            if len(similar_genres) == top_n:
                break

        return pd.DataFrame(similar_genres), None

    def recommend_by_genre(self, song_name, artist_name=None, top_n=5):
        """
        Given a song name and optional artist:
        1. Deterministically resolves the track in the dataset.
        2. Retrieves the song's genre.
        3. Identifies candidate songs belonging to the exact same genre.
        4. Calculates Cosine Similarity of audio features against the query song.
        5. Ranks deterministically and returns top_n most similar tracks within that genre.
        """
        if self.df is None:
            self.load_data_and_models()

        song_meta, err = find_song(self.df, song_name, artist_name)
        if err:
            return None, err

        query_genre = song_meta["track_genre"]

        print(f"\n[Search Match] Found song: '{song_meta['track_name']}' by {song_meta['artists']}")
        print(f"[GenreRecommender] Song genre: '{query_genre}' (Popularity: {song_meta.get('popularity', 'N/A')})")
        print(f"[GenreRecommender] Filtering other tracks in genre '{query_genre}' and ranking...")

        # Filter dataset for tracks with the same genre
        genre_df = self.df[self.df["track_genre"] == query_genre].copy()

        # Remove the query track itself
        genre_df = genre_df[genre_df["track_id"] != song_meta["track_id"]]

        if genre_df.empty:
            return None, f"No other tracks found in the genre '{query_genre}' for recommendations."

        # Extract audio features for similarity calculation
        # Retrieve the complete original row because find_song()
        # returns a metadata-only formatted row.
        query_track_id = song_meta["track_id"]

        query_rows = self.df[
            self.df["track_id"] == query_track_id
        ]

        if query_rows.empty:
            return None, (
                f"Could not retrieve full feature data for "
                f"track_id '{query_track_id}'."
            )

        query_full_row = query_rows.iloc[0]

        # Make sure scaled audio features exist
        if not self.scaled_feature_cols:
            return None, (
                "No scaled audio feature columns found in the dataset. "
                "Expected columns beginning with 'scaled_'."
            )

        query_features = (
            query_full_row[self.scaled_feature_cols]
            .values
            .astype(float)
            .reshape(1, -1)
        )        
        candidate_features = genre_df[self.scaled_feature_cols].values.astype(float)

        similarities = cosine_similarity(query_features, candidate_features)[0]
        genre_df["similarity_score"] = similarities

        # Deterministic sorting
        sort_cols = ["similarity_score"]
        ascending_flags = [False]
        if "popularity" in genre_df.columns:
            sort_cols.append("popularity")
            ascending_flags.append(False)
        sort_cols.append("track_id")
        ascending_flags.append(True)

        rec_df = genre_df.sort_values(by=sort_cols, ascending=ascending_flags).head(top_n)

        output_cols = [
            c for c in ["track_name", "artists", "album_name", "track_genre", "popularity", "similarity_score"]
            if c in rec_df.columns
        ]
        return rec_df[output_cols].reset_index(drop=True), None


# ==============================================================================
# Package-level Wrapper Function
# ==============================================================================

_global_genre_recommender = None


def recommend_by_genre(song_name, artist_name=None, top_n=5):
    """
    Module level helper function matching the recommend_by_genre requirement.
    Lazily initializes the global GenreRecommender instance.
    """
    global _global_genre_recommender
    if _global_genre_recommender is None:
        _global_genre_recommender = GenreRecommender()

    return _global_genre_recommender.recommend_by_genre(song_name, artist_name, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("           GENRE RECOMMENDER SYSTEM TESTING BLOCK           ")
    print("=" * 60)

    recommender = GenreRecommender()
    try:
        recommender.load_data_and_models()

        test_genre = "hardcore"
        print(f"\n--- Finding similar genres to '{test_genre}' ---")
        similar_df, err = recommender.get_similar_genres(test_genre, top_n=5)
        if err:
            print(f"Error: {err}")
        else:
            print(similar_df.to_string(index=False))

        test_song = "Blinding Lights"
        print(f"\n--- Running recommend_by_genre for '{test_song}' ---")
        rec_df, err = recommend_by_genre(test_song, top_n=5)
        if err:
            print(f"Error: {err}")
        else:
            print(rec_df.to_string(index=False))
    except Exception as e:
        print(f"\n[Execution Error] {e}")
    print("=" * 60)
