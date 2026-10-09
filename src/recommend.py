import os
import sys

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.song_lookup import find_song


class ContentBasedRecommender:
    def __init__(self, processed_df):
        """
        Initializes the recommender with a preprocessed DataFrame.
        Expected to have 'track_name', 'artists', and scaled feature columns.
        """
        self.df = processed_df.reset_index(drop=True)
        # Extract columns starting with 'scaled_' as our feature matrix
        self.feature_cols = [
            col for col in self.df.columns if col.startswith("scaled_")
        ]
        self.features = self.df[self.feature_cols].values.astype(float)
        self.similarity_matrix = None

    def fit(self, build_matrix=False):
        """
        Prepares the recommender. If build_matrix=True, also precomputes and stores
        the pairwise cosine similarity matrix for the top tracks.
        """
        print("Preparing audio features for similarity engine...")
        if build_matrix:
            self.build_similarity_matrix()
            print("Cosine similarity matrix built and stored in memory.")
        print("Model fitted successfully.")

    def build_similarity_matrix(self, max_tracks=5000):
        """
        Computes the cosine similarity matrix for the top max_tracks (sorted by popularity)
        to prevent excessive memory usage while precomputing artifacts.
        Recommendations on arbitrary tracks are calculated on-the-fly with high performance.
        """
        num_tracks = len(self.features)
        if num_tracks > max_tracks:
            print(
                f"[Info] Dataset has {num_tracks} tracks. Precomputing similarity matrix "
                f"for top {max_tracks} most popular tracks to balance memory footprint."
            )
            if "popularity" in self.df.columns:
                top_indices = (
                    self.df.sort_values("popularity", ascending=False)
                    .head(max_tracks)
                    .index
                )
                subset_features = self.features[top_indices]
            else:
                subset_features = self.features[:max_tracks]
            self.similarity_matrix = cosine_similarity(subset_features)
        else:
            self.similarity_matrix = cosine_similarity(self.features)
        return self.similarity_matrix

    def save_similarity_matrix(self, output_path):
        """
        Saves the precomputed cosine similarity matrix to disk using pickle.
        """
        if self.similarity_matrix is None:
            raise ValueError(
                "Similarity matrix has not been built yet. Call build_similarity_matrix() first."
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.similarity_matrix, f)
        print(f"Cosine similarity matrix saved to {output_path}")

    def get_song_recommendations(self, query_song, artist_name=None, top_n=5):
        """
        Recommends the top_n most similar songs to the query_song.
        Supports optional artist_name for unambiguous track resolution.
        Uses deterministic candidate ranking and eliminates row-order bias.
        """
        # Resolve track deterministically using song_lookup
        song_meta, err = find_song(self.df, query_song, artist_name)
        if err:
            return None, err

        print(
            f"\nFound song: '{song_meta['track_name']}' by {song_meta['artists']} "
            f"(Popularity: {song_meta.get('popularity', 'N/A')})"
        )
        print("Calculating recommendations...")

        # Extract the query song's feature vector and reshape to (1, D)
        query_vector = song_meta[self.feature_cols].values.astype(float).reshape(1, -1)

        # Compute cosine similarity between this song and all songs in dataset on-the-fly
        similarity_scores = cosine_similarity(query_vector, self.features)[0]

        # Build results DataFrame
        results_df = self.df.copy()
        results_df["similarity_score"] = similarity_scores

        # Filter out the query track itself by track_id
        results_df = results_df[results_df["track_id"] != song_meta["track_id"]]

        # Deterministic sort: similarity descending -> popularity descending -> track_id ascending
        sort_cols = ["similarity_score"]
        ascending_flags = [False]
        if "popularity" in results_df.columns:
            sort_cols.append("popularity")
            ascending_flags.append(False)
        sort_cols.append("track_id")
        ascending_flags.append(True)

        top_candidates = results_df.sort_values(by=sort_cols, ascending=ascending_flags).head(top_n)

        recommendations = []
        for _, rec_song in top_candidates.iterrows():
            recommendations.append(
                {
                    "track_name": rec_song["track_name"],
                    "artists": rec_song["artists"],
                    "album_name": rec_song.get("album_name", "N/A"),
                    "similarity_score": rec_song["similarity_score"],
                }
            )

        rec_df = pd.DataFrame(recommendations)
        return rec_df, None
