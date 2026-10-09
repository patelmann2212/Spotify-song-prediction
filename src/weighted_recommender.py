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
from src.kmeans_recommender import KMeansRecommender


DEFAULT_HYBRID_WEIGHTS = {
    "audio": 0.40,
    "cluster": 0.15,
    "artist": 0.20,
    "genre": 0.15,
    "popularity": 0.10,
}


class WeightedRecommender:
    """
    Weighted Hybrid Recommendation Engine.

    Combines five distinct signals:
    1. Audio Feature Similarity (40% default weight):
       - Cosine similarity across scaled audio features.
    2. K-Means Cluster Audio Match (15% default weight):
       - Audio cosine similarity if in the same audio cluster, 0.0 otherwise.
    3. Artist Name Similarity (20% default weight):
       - TF-IDF name overlap similarity of artists.
    4. Genre Profile Similarity (15% default weight):
       - Cosine similarity of average audio profiles of song genres.
    5. Popularity Score (10% default weight):
       - Normalized popularity value.

    Formula:
    final_score = (
        w_audio * audio_sim +
        w_cluster * cluster_sim +
        w_artist * artist_sim +
        w_genre * genre_sim +
        w_popularity * popularity
    )
    """
    def __init__(self, cleaned_data_path=None, model_dir="models", weights=None):
        if cleaned_data_path is None:
            if os.path.exists("data/cleaned_spotify.csv"):
                cleaned_data_path = "data/cleaned_spotify.csv"
            else:
                cleaned_data_path = "data/processed/train.csv"

        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir
        self.weights = dict(weights) if weights else dict(DEFAULT_HYBRID_WEIGHTS)

        self.audio_engine = None
        self.artist_engine = None
        self.genre_engine = None
        self.kmeans_engine = None

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

        # 4. K-Means Recommender Engine
        try:
            self.kmeans_engine = KMeansRecommender(
                df=self.df,
                cleaned_data_path=self.cleaned_data_path,
                model_dir=self.model_dir
            )
            self.kmeans_engine.fit()
        except Exception as e:
            print(f"[WeightedRecommender] Warning: Could not initialize KMeansRecommender ({e}).")
            self.kmeans_engine = None

        print("[WeightedRecommender] All sub-engines loaded successfully.\n")

    def get_weighted_recommendations(
        self,
        song_name,
        artist_name=None,
        top_n=10,
        weights=None,
        filter_by_cluster=True,
    ):
        """
        Calculates the hybrid recommendation score relative to the query song,
        using K-Means to filter candidates to the same cluster, sorts deterministically,
        and returns top_n tracks.
        """
        if self.df is None:
            self.load_engines()

        active_weights = dict(weights) if weights else self.weights

        # 1. Deterministic Song Lookup / Hybrid Search
        if isinstance(song_name, (pd.Series, dict)):
            query_song = song_name
        else:
            query_song, err = find_song(self.df, song_name, artist_name)
            if err:
                return None, err

        print(f"\n[Hybrid] Found query song: '{query_song['track_name']}' by {query_song['artists']}")
        print(f"[Hybrid] Genre: '{query_song.get('track_genre', 'N/A')}' | Popularity: {query_song.get('popularity', 'N/A')}")

        # 2. Retrieve complete original row from self.df using track_id
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

        # 3. K-Means Candidate Filtering
        # Predict cluster and narrow the candidate search space strictly to same-cluster songs
        query_cluster = None
        if self.kmeans_engine is not None and filter_by_cluster:
            try:
                query_cluster = self.kmeans_engine.predict_cluster(query_full_row)
                candidate_clusters = self.kmeans_engine.cluster_df["cluster"].values
                same_cluster_indices = self.kmeans_engine.cluster_df.index[candidate_clusters == query_cluster]
                candidate_pool = self.df.loc[same_cluster_indices].copy()
                print(f"[Hybrid] K-Means Cluster: {query_cluster} ({len(candidate_pool)} candidate tracks in cluster)")
            except Exception as e:
                print(f"[Hybrid Warning] Failed to filter candidates by cluster: {e}")
                candidate_pool = self.df.copy()
        else:
            candidate_pool = self.df.copy()

        # Remove the selected song itself from the candidate pool
        candidate_pool = candidate_pool[candidate_pool["track_id"] != query_track_id]
        if candidate_pool.empty:
            return pd.DataFrame(), None

        print("[Hybrid] Calculating recommendation signals on candidate pool...")

        # 4. Audio Feature Similarity (Cosine Similarity on scaled audio features)
        query_audio = (
            query_full_row[self.scaled_feature_cols]
            .values
            .astype(float)
            .reshape(1, -1)
        )
        candidate_audios = (
            candidate_pool[self.scaled_feature_cols]
            .values
            .astype(float)
        )
        audio_similarities = cosine_similarity(
            query_audio,
            candidate_audios
        )[0]

        # 5. K-Means Cluster Audio Match
        # (For same-cluster candidates, cluster similarity is the audio similarity; 0.0 for cross-cluster)
        if query_cluster is not None and self.kmeans_engine is not None:
            cand_clusters = self.kmeans_engine.cluster_df.loc[candidate_pool.index, "cluster"].values
            same_cluster_mask = (cand_clusters == query_cluster)
            cluster_similarities = np.where(same_cluster_mask, audio_similarities, 0.0)
        else:
            cluster_similarities = audio_similarities

        # 6. Artist Name Similarity (TF-IDF token overlap)
        query_artist_str = str(query_song["artists"])
        query_artist_vec = self.artist_engine.vectorizer.transform([query_artist_str])
        candidate_artists_vecs = self.artist_engine.vectorizer.transform(candidate_pool["artists"].astype(str))
        artist_similarities = cosine_similarity(query_artist_vec, candidate_artists_vecs)[0]

        # 7. Genre Profile Similarity
        query_genre = str(query_song.get("track_genre", ""))
        if query_genre in self.genre_engine.genre_to_idx:
            query_genre_idx = self.genre_engine.genre_to_idx[query_genre]
            genre_sims_by_idx = self.genre_engine.genre_similarity_matrix[query_genre_idx]
            genre_sim_map = {
                self.genre_engine.unique_genres[i]: genre_sims_by_idx[i]
                for i in range(len(self.genre_engine.unique_genres))
            }
            genre_similarities = candidate_pool["track_genre"].map(genre_sim_map).fillna(0.0).values
        else:
            genre_similarities = np.zeros(len(candidate_pool))

        # 8. Normalized Popularity Score
        pop_scaler = MinMaxScaler()
        if "popularity" in self.df.columns:
            pop_scaler.fit(self.df[["popularity"]])
            pop_scores = pop_scaler.transform(candidate_pool[["popularity"]]).flatten()
        else:
            pop_scores = np.zeros(len(candidate_pool))

        # 9. Combined Weighted Score
        w = active_weights
        final_scores = (
            w.get("audio", 0.40) * audio_similarities +
            w.get("cluster", 0.15) * cluster_similarities +
            w.get("artist", 0.20) * artist_similarities +
            w.get("genre", 0.15) * genre_similarities +
            w.get("popularity", 0.10) * pop_scores
        )

        # 10. Assemble Results
        results_df = candidate_pool.copy()
        results_df["audio_similarity"] = audio_similarities
        results_df["cluster_similarity"] = cluster_similarities
        if self.kmeans_engine is not None and self.kmeans_engine.cluster_df is not None:
            results_df["cluster_id"] = self.kmeans_engine.cluster_df.loc[candidate_pool.index, "cluster"].values
        results_df["artist_similarity"] = artist_similarities
        results_df["genre_similarity"] = genre_similarities
        results_df["popularity_score"] = pop_scores
        results_df["final_score"] = final_scores

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


def get_weighted_recommendations(
    song_name, artist_name=None, top_n=10, weights=None, filter_by_cluster=True
):
    """
    Module level helper function matching get_weighted_recommendations requirement.
    Lazily initializes the global WeightedRecommender instance.
    """
    global _global_weighted_recommender
    if _global_weighted_recommender is None:
        _global_weighted_recommender = WeightedRecommender()

    return _global_weighted_recommender.get_weighted_recommendations(
        song_name, artist_name, top_n, weights=weights, filter_by_cluster=filter_by_cluster
    )


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
                    "track_name", "artists", "audio_similarity", "cluster_similarity",
                    "artist_similarity", "genre_similarity", "popularity_score", "final_score"
                ] if c in rec_df.columns
            ]
            print(rec_df[cols_to_print].to_string(index=False))
    except Exception as e:
        print(f"\n[Execution Error] {e}")
    print("=" * 60)
