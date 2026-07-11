import os
import sys

# Ensure the root of the project is in python path to handle relative module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Import our existing recommender engines to reuse their features
from src.recommend import ContentBasedRecommender
from src.genre_recommender import GenreRecommender
from src.artist_recommender import ArtistSimilarityRecommender

class WeightedRecommender:
    """
    Weighted Hybrid Recommendation Engine.
    
    This engine combines four distinct signals to rank songs:
    1. Audio Feature Similarity (50% weight):
       - Cosine similarity of the 10 scaled audio features.
    2. Artist Name Similarity (20% weight):
       - TF-IDF name overlap similarity of artists.
    3. Genre Profile Similarity (20% weight):
       - Cosine similarity of the average audio profiles of the song genres.
    4. Popularity Score (10% weight):
       - Normalized popularity value.
       
    Formula:
    final_score = (0.50 * audio_sim) + (0.20 * artist_sim) + (0.20 * genre_sim) + (0.10 * popularity)
    """
    def __init__(self, cleaned_data_path="data/processed/train.csv", model_dir="models"):
        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir
        
        # Load underlying engines
        self.audio_engine = None
        self.artist_engine = None
        self.genre_engine = None
        
        # Unified DataFrame
        self.df = None
        self.scaled_feature_cols = None

    def load_engines(self):
        """
        Initializes and loads the underlying recommendation systems.
        """
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Training dataset not found at {self.cleaned_data_path}. Please run main.py first."
            )
            
        print("[WeightedRecommender] Initializing and loading sub-engines...")
        self.df = pd.read_csv(self.cleaned_data_path)
        self.scaled_feature_cols = [col for col in self.df.columns if col.startswith("scaled_")]
        
        # 1. Load Audio Feature Engine
        self.audio_engine = ContentBasedRecommender(self.df)
        self.audio_engine.fit(build_matrix=False) # We calculate similarity on-the-fly for the query song
        
        # 2. Load Genre Recommender Engine
        self.genre_engine = GenreRecommender(cleaned_data_path=self.cleaned_data_path, model_dir=self.model_dir)
        self.genre_engine.load_data_and_models()
        
        # 3. Load Artist Recommender Engine
        self.artist_engine = ArtistSimilarityRecommender(cleaned_data_path=self.cleaned_data_path, model_dir=self.model_dir)
        self.artist_engine.load_data_and_models()
        
        print("[WeightedRecommender] All sub-engines loaded successfully.\n")

    def get_weighted_recommendations(self, song_name, top_n=10):
        """
        Calculates the hybrid recommendation score for all songs in the dataset
        relative to the query song, sorts them, and returns the top_n tracks.
        """
        # Ensure all sub-engines are fully loaded
        if self.df is None:
            self.load_engines()
            
        # 1. LOOK UP QUERY SONG
        # Case-insensitive check
        matches = self.df[self.df["track_name"].str.lower() == song_name.lower()]
        if matches.empty:
            # Fallback to partial search
            matches = self.df[self.df["track_name"].str.lower().str.contains(song_name.lower(), regex=False)]
            
        if matches.empty:
            return None, f"No song matching '{song_name}' found in the dataset."
            
        # Select the most popular match as our query track
        query_idx = self.df.index.get_loc(matches.index[0])
        query_song = self.df.iloc[query_idx]
        
        print(f"\n[Hybrid] Found query song: '{query_song['track_name']}' by {query_song['artists']}")
        print(f"[Hybrid] Genre: '{query_song['track_genre']}' | Popularity: {query_song['popularity']}")
        print("[Hybrid] Calculating similarities and ranking all candidate songs...")

        # 2. SIGNAL 1: AUDIO FEATURE SIMILARITY (50% Weight)
        # Extract the scaled features and calculate Cosine Similarity against all tracks
        query_audio = query_song[self.scaled_feature_cols].values.reshape(1, -1)
        candidate_audios = self.df[self.scaled_feature_cols].values
        audio_similarities = cosine_similarity(query_audio, candidate_audios)[0]

        # 3. SIGNAL 2: ARTIST NAME SIMILARITY (20% Weight)
        # Vectorize the query artist name and all dataset artists, then calculate similarity
        query_artist_str = query_song["artists"]
        query_artist_vector = self.artist_engine.vectorizer.transform([query_artist_str])
        dataset_artists_vectors = self.artist_engine.vectorizer.transform(self.df["artists"])
        artist_similarities = cosine_similarity(query_artist_vector, dataset_artists_vectors)[0]

        # 4. SIGNAL 3: GENRE PROFILE SIMILARITY (20% Weight)
        # Get the query song's genre and retrieve its similarity to all other genres in the database
        query_genre = query_song["track_genre"]
        query_genre_idx = self.genre_engine.genre_to_idx[query_genre]
        genre_similarities_by_idx = self.genre_engine.genre_similarity_matrix[query_genre_idx]
        
        # Create a map from genre names to their similarity scores relative to the query genre
        genre_sim_map = {
            self.genre_engine.unique_genres[i]: genre_similarities_by_idx[i]
            for i in range(len(self.genre_engine.unique_genres))
        }
        # Apply the map to every track in the dataset
        genre_similarities = self.df["track_genre"].map(genre_sim_map).values

        # 5. SIGNAL 4: NORMALIZED POPULARITY SCORE (10% Weight)
        # Scale the popularity of all tracks in the database to a [0, 1] range using Min-Max scaling
        scaler = MinMaxScaler()
        popularity_scores = scaler.fit_transform(self.df[["popularity"]]).flatten()

        # 6. COMBINE SIGNALS (WEIGHTED FORMULA)
        final_scores = (
            0.50 * audio_similarities +
            0.20 * artist_similarities +
            0.20 * genre_similarities +
            0.10 * popularity_scores
        )

        # 7. ASSEMBLE RESULTS AND SORT
        # Create a copy of the dataset and attach the score breakdowns
        results_df = self.df.copy()
        results_df["audio_similarity"] = audio_similarities
        results_df["artist_similarity"] = artist_similarities
        results_df["genre_similarity"] = genre_similarities
        results_df["popularity_score"] = popularity_scores
        results_df["final_score"] = final_scores

        # Filter out the query track itself so we don't recommend a song to itself
        results_df = results_df[results_df["track_id"] != query_song["track_id"]]

        # Sort tracks by final hybrid score in descending order (highest first)
        top_recommendations = results_df.sort_values(by="final_score", ascending=False).head(top_n)
        top_recommendations = top_recommendations.reset_index(drop=True)
        
        return top_recommendations, None


# ==============================================================================
# Package-level Wrapper Function (Required by prompt)
# ==============================================================================

_global_weighted_recommender = None

def get_weighted_recommendations(song_name, top_n=10):
    """
    Module level helper function matching the get_weighted_recommendations requirement.
    Lazily initializes the global WeightedRecommender instance.
    """
    global _global_weighted_recommender
    if _global_weighted_recommender is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "processed", "train.csv")
        model_dir = os.path.join(base_dir, "models")
        _global_weighted_recommender = WeightedRecommender(cleaned_data_path=train_path, model_dir=model_dir)
        
    return _global_weighted_recommender.get_weighted_recommendations(song_name, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("         WEIGHTED RECOMMENDER TESTING BLOCK          ")
    print("=" * 60)
    
    recommender = WeightedRecommender()
    try:
        # Load and set up engines
        recommender.load_engines()
        
        # Test query
        test_song = "Blinding Lights"
        print(f"Querying recommendations for '{test_song}'...")
        
        rec_df, err = recommender.get_weighted_recommendations(test_song, top_n=10)
        
        if err:
            # Fallback to first song in the database if "Blinding Lights" is missing
            fallback_song = recommender.df.iloc[0]["track_name"]
            print(f"'{test_song}' not found, trying fallback: '{fallback_song}'")
            rec_df, err = recommender.get_weighted_recommendations(fallback_song, top_n=10)
            
        if err:
            print(f"Error: {err}")
        else:
            # Print output table
            print("\nTop 10 Hybrid Recommendations:")
            cols_to_print = ["track_name", "artists", "audio_similarity", "artist_similarity", "genre_similarity", "popularity_score", "final_score"]
            print(rec_df[cols_to_print].to_string(index=False))
            
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        print("Note: Ensure data/processed/train.csv exists by running main.py first.")
    print("=" * 60)
