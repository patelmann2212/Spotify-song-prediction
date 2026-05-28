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
        self.feature_cols = [col for col in self.df.columns if col.startswith("scaled_")]
        self.features = self.df[self.feature_cols].values
        self.similarity_matrix = None
        
    def fit(self):
        """
        Prepares the recommender. To support large datasets (114,000+ tracks) without
        MemoryErrors, we skip pre-computing the full N x N matrix and compute similarity
        dynamically on-the-fly.
        """
        print("Preparing audio features for similarity engine...")
        print("Model fitted successfully.")
        
    def get_song_recommendations(self, query_song, top_n=5):
        """
        Recommends the top_n most similar songs to the query_song (case-insensitive).
        Returns a DataFrame of recommendations with similarity scores.
        """
        # Case-insensitive lookup for matching songs
        matches = self.df[self.df["track_name"].str.lower() == query_song.lower()]
        
        # If no exact match, try partial match
        if matches.empty:
            matches = self.df[self.df["track_name"].str.lower().str.contains(query_song.lower(), regex=False)]
            
        if matches.empty:
            return None, f"No song matching '{query_song}' found in the dataset."
            
        # If there are multiple matches, take the first one (highest popularity due to preprocessing sorting)
        song_idx = matches.index[0]
        song_meta = self.df.iloc[song_idx]
        
        print(f"\nFound song: '{song_meta['track_name']}' by {song_meta['artists']} "
              f"(Popularity: {song_meta.get('popularity', 'N/A')})")
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
            
            recommendations.append({
                "track_name": rec_song["track_name"],
                "artists": rec_song["artists"],
                "album_name": rec_song.get("album_name", "N/A"),
                "similarity_score": similarity
            })
            
            if len(recommendations) == top_n:
                break
                
        rec_df = pd.DataFrame(recommendations)
        return rec_df, None
