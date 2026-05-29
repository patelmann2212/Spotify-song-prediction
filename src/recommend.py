import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, processed_df):
        """
        Initializes the recommender with a preprocessed DataFrame.
        Expected to have 'track_name', 'artists', and scaled feature columns.
        """
        self.df = processed_df
        # Extract columns starting with 'scaled_' as our feature matrix
        self.feature_cols = [
            col for col in self.df.columns if col.startswith("scaled_")
        ]
        self.features = self.df[self.feature_cols].values
        self.similarity_matrix = None

    def fit(self, build_matrix=False):
        """
        Prepares the recommender. If build_matrix=True, also precomputes and stores
        the pairwise cosine similarity matrix for the dataset.
        """
        print("Preparing audio features for similarity engine...")
        if build_matrix:
            self.build_similarity_matrix()
            print("Cosine similarity matrix built and stored in memory.")
        print("Model fitted successfully.")

    def build_similarity_matrix(self, max_tracks=5000):
        """
        Computes the cosine similarity matrix. For large datasets, we precompute
        similarity for the top max_tracks (sorted by popularity) to prevent MemoryErrors (31.6 GiB),
        while still supporting high-performance on-the-fly calculation for recommendations.
        """
        num_tracks = len(self.features)
        if num_tracks > max_tracks:
            print(f"[Warning] Dataset is too large ({num_tracks} tracks) to precompute the full similarity matrix (requires ~31.6 GiB).")
            print(f"-> Precomputing similarity matrix for the top {max_tracks} most popular tracks to prevent MemoryErrors.")
            
            if "popularity" in self.df.columns:
                # Sort features to get top popular tracks
                top_indices = self.df.sort_values("popularity", ascending=False).head(max_tracks).index
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

    def get_song_recommendations(self, query_song, top_n=5):
        """
        Recommends the top_n most similar songs to the query_song (case-insensitive).
        Returns a DataFrame of recommendations with similarity scores.
        """
        # Case-insensitive lookup for matching songs
        matches = self.df[self.df["track_name"].str.lower() == query_song.lower()]

        # If no exact match, try partial match
        if matches.empty:
            matches = self.df[
                self.df["track_name"]
                .str.lower()
                .str.contains(query_song.lower(), regex=False)
            ]

        if matches.empty:
            return None, f"No song matching '{query_song}' found in the dataset."

        # If there are multiple matches, take the first one (highest popularity due to preprocessing sorting)
        # We find its positional integer index in self.df to index self.features properly
        song_idx = self.df.index.get_loc(matches.index[0])
        song_meta = self.df.iloc[song_idx]

        print(
            f"\nFound song: '{song_meta['track_name']}' by {song_meta['artists']} "
            f"(Popularity: {song_meta.get('popularity', 'N/A')})"
        )
        print("Calculating recommendations...")

        # Extract the query song's feature vector and reshape to (1, D)
        query_vector = self.features[song_idx].reshape(1, -1)

        # Compute cosine similarity between this song and all other songs in the dataset on-the-fly
        # Returns shape (1, N), we slice the first row [0] to get a 1D array of shape (N,)
        similarity_scores = cosine_similarity(query_vector, self.features)[0]

        # Sort indices in descending order (highest similarity first)
        similar_indices = np.argsort(similarity_scores)[::-1]

        # Filter out the query song itself from recommendations
        recommendations = []
        for idx in similar_indices:
            if idx == song_idx:
                continue

            similarity = similarity_scores[idx]
            rec_song = self.df.iloc[idx]

            recommendations.append(
                {
                    "track_name": rec_song["track_name"],
                    "artists": rec_song["artists"],
                    "album_name": rec_song.get("album_name", "N/A"),
                    "similarity_score": similarity,
                }
            )

            if len(recommendations) == top_n:
                break

        rec_df = pd.DataFrame(recommendations)
        return rec_df, None
