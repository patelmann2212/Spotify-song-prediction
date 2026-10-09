import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Ensure root of project is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.song_lookup import find_song
from src.recommend import ContentBasedRecommender
from src.genre_recommender import GenreRecommender
from src.artist_recommender import ArtistSimilarityRecommender


class WeightedRecommender:
    """
    Weighted Hybrid Recommendation Engine.

    Combines four distinct signals:
    1. Audio Feature Similarity (50% weight):
       - Cosine similarity across scaled audio features.
    2. Artist Name Similarity (20% weight):
       - TF-IDF name overlap similarity of artists.
    3. Genre Profile Similarity (20% weight):
       - Cosine similarity of average audio profiles of song genres.
    4. Popularity Score (10% weight):
       - Normalized popularity value.

    Formula:
    final_score = (0.50 * audio_sim) + (0.20 * artist_sim) + (0.20 * genre_sim) + (0.10 * popularity)
    """
    def __init__(self, cleaned_data_path=None, model_dir="models"):
        if cleaned_data_path is None:
            if os.path.exists("data/cleaned_spotify.csv"):
                cleaned_data_path = "data/cleaned_spotify.csv"
            else:
                cleaned_data_path = "data/processed/train.csv"

        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir

        self.audio_engine = None
        self.artist_engine = None
        self.genre_engine = None

        self.df = None
        self.scaled_feature_cols = None

    def load_engines(self):
        """
        Initializes and loads the underlying recommendation systems.
        """
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.cleaned_data_path}. Please run preprocess.py first."
            )

        print("[WeightedRecommender] Initializing and loading sub-engines...")
        self.df = pd.read_csv(self.cleaned_data_path).reset_index(drop=True)
        self.scaled_feature_cols = [
            col for col in self.df.columns if col.startswith("scaled_")
        ]

        # 1. Audio Feature Engine
        self.audio_engine = ContentBasedRecommender(self.df)
        self.audio_engine.fit(build_matrix=False)

        # 2. Genre Recommender Engine
        self.genre_engine = GenreRecommender(
            cleaned_data_path=self.cleaned_data_path,
            model_dir=self.model_dir
        )
        self.genre_engine.load_data_and_models()

        # 3. Artist Recommender Engine
        self.artist_engine = ArtistSimilarityRecommender(
            cleaned_data_path=self.cleaned_data_path,
            model_dir=self.model_dir
        )
        self.artist_engine.load_data_and_models()

        print("[WeightedRecommender] All sub-engines loaded successfully.\n")

    def get_weighted_recommendations(self, song_name, artist_name=None, top_n=10):
        """
        Calculates the hybrid recommendation score for all songs in the dataset
        relative to the query song, sorts deterministically, and returns top_n tracks.
        """
        if self.df is None:
            self.load_engines()

        # 1. Deterministic Song Lookup
        query_song, err = find_song(self.df, song_name, artist_name)
        if err:
            return None, err

        print(f"\n[Hybrid] Found query song: '{query_song['track_name']}' by {query_song['artists']}")
        print(f"[Hybrid] Genre: '{query_song.get('track_genre', 'N/A')}' | Popularity: {query_song.get('popularity', 'N/A')}")
        print("[Hybrid] Calculating similarities and ranking candidate songs...")

        # 2. Audio Feature Similarity (50% Weight)
        #
        # find_song() returns a formatted Series containing metadata columns,
        # so it may not contain the scaled audio-feature columns.
        # Retrieve the complete original row from self.df using track_id.

        query_track_id = query_song["track_id"]

        query_rows = self.df[self.df["track_id"] == query_track_id]

        if query_rows.empty:
            return None, f"Could not retrieve full feature data for track_id '{query_track_id}'."

        query_full_row = query_rows.iloc[0]

        # Validate that scaled audio features are available
        if not self.scaled_feature_cols:
            return None, (
                "No scaled audio feature columns found in the dataset. "
                "Expected columns such as scaled_energy, scaled_danceability, "
                "scaled_valence, etc."
            )

        # Extract complete query feature vector
        query_audio = (
            query_full_row[self.scaled_feature_cols]
            .values
            .astype(float)
            .reshape(1, -1)
        )

        # Extract candidate feature matrix
        candidate_audios = (
            self.df[self.scaled_feature_cols]
            .values
            .astype(float)
        )

        audio_similarities = cosine_similarity(
            query_audio,
            candidate_audios
        )[0]
        
        # 3. Artist Name Similarity (20% Weight)
        query_artist_str = str(query_song["artists"])
        query_artist_vec = self.artist_engine.vectorizer.transform([query_artist_str])
        dataset_artists_vecs = self.artist_engine.vectorizer.transform(self.df["artists"].astype(str))
        artist_similarities = cosine_similarity(query_artist_vec, dataset_artists_vecs)[0]

        # 4. Genre Profile Similarity (20% Weight)
        query_genre = str(query_song.get("track_genre", ""))
        if query_genre in self.genre_engine.genre_to_idx:
            query_genre_idx = self.genre_engine.genre_to_idx[query_genre]
            genre_sims_by_idx = self.genre_engine.genre_similarity_matrix[query_genre_idx]
            genre_sim_map = {
                self.genre_engine.unique_genres[i]: genre_sims_by_idx[i]
                for i in range(len(self.genre_engine.unique_genres))
            }
            genre_similarities = self.df["track_genre"].map(genre_sim_map).fillna(0.0).values
        else:
            genre_similarities = np.zeros(len(self.df))

        # 5. Normalized Popularity Score (10% Weight)
        pop_scaler = MinMaxScaler()
        pop_scores = pop_scaler.fit_transform(self.df[["popularity"]]).flatten()

        # 6. Combined Weighted Score
        final_scores = (
            0.50 * audio_similarities +
            0.20 * artist_similarities +
            0.20 * genre_similarities +
            0.10 * pop_scores
        )

        # 7. Assemble Results
        results_df = self.df.copy()
        results_df["audio_similarity"] = audio_similarities
        results_df["artist_similarity"] = artist_similarities
        results_df["genre_similarity"] = genre_similarities
        results_df["popularity_score"] = pop_scores
        results_df["final_score"] = final_scores

        # Filter out the query track itself by track_id
        results_df = results_df[results_df["track_id"] != query_song["track_id"]]

        # Sort deterministically
        sort_cols = ["final_score", "popularity", "track_id"]
        top_recommendations = results_df.sort_values(
            by=sort_cols, ascending=[False, False, True]
        ).head(top_n)

        return top_recommendations.reset_index(drop=True), None


# ==============================================================================
# Package-level Wrapper Function
# ==============================================================================

_global_weighted_recommender = None


def get_weighted_recommendations(song_name, artist_name=None, top_n=10):
    """
    Module level helper function matching get_weighted_recommendations requirement.
    Lazily initializes the global WeightedRecommender instance.
    """
    global _global_weighted_recommender
    if _global_weighted_recommender is None:
        _global_weighted_recommender = WeightedRecommender()

    return _global_weighted_recommender.get_weighted_recommendations(song_name, artist_name, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("         WEIGHTED RECOMMENDER TESTING BLOCK          ")
    print("=" * 60)

    recommender = WeightedRecommender()
    try:
        recommender.load_engines()

        test_song = "Blinding Lights"
        print(f"Querying recommendations for '{test_song}'...")

        rec_df, err = recommender.get_weighted_recommendations(test_song, top_n=10)
        if err:
            print(f"Error: {err}")
        else:
            cols_to_print = [
                c for c in [
                    "track_name", "artists", "audio_similarity",
                    "artist_similarity", "genre_similarity", "popularity_score", "final_score"
                ] if c in rec_df.columns
            ]
            print(rec_df[cols_to_print].to_string(index=False))
    except Exception as e:
        print(f"\n[Execution Error] {e}")
    print("=" * 60)
