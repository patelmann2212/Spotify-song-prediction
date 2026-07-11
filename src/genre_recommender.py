import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

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
       - Ranks same-genre candidate songs by their audio similarity to the query song.
    """
    def __init__(self, cleaned_data_path="data/processed/train.csv", model_dir="models"):
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
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # 1. Load the underlying cleaned training dataset
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Training dataset not found at {self.cleaned_data_path}. Please run main.py or preprocess.py first."
            )
        
        print(f"[GenreRecommender] Loading dataset from {self.cleaned_data_path}...")
        self.df = pd.read_csv(self.cleaned_data_path)
        
        # Identify features starting with 'scaled_' for similarity calculations
        self.scaled_feature_cols = [col for col in self.df.columns if col.startswith("scaled_")]
        
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

        # 2. Build model if not cached
        print("[GenreRecommender] Encoding genres and calculating similarities...")
        self.encode_genres()
        self.calculate_genre_similarity()
        self.save_model()

    def encode_genres(self):
        """
        Performs two encodings of the track genres:
        1. Song-Level One-Hot Encoding: Creates binary vectors for genres using scikit-learn.
        2. Genre-Level Audio Profiling: Computes a mean audio vector for each genre from training tracks.
        """
        # Ensure track_genre exists and is clean
        if "track_genre" not in self.df.columns:
            raise KeyError("The 'track_genre' column is missing from the dataset.")
        
        self.df["track_genre"] = self.df["track_genre"].astype(str).str.strip()

        # --- Encoding 1: Categorical One-Hot Encoding ---
        # Fits an encoder that represents genres as 1-of-K arrays
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Reshape to 2D array as required by scikit-learn
        self.one_hot_encoder.fit(self.df[["track_genre"]])
        
        # --- Encoding 2: Genre-Level Audio Profiling ---
        # Group by genre and compute the mean of scaled audio features
        genre_profiles_df = self.df.groupby("track_genre")[self.scaled_feature_cols].mean()
        
        # Keep track of unique genres and construct a fast lookup map
        self.unique_genres = genre_profiles_df.index.tolist()
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.unique_genres)}
        
        # Store profiles as a numpy array of shape (num_genres, num_audio_features)
        self.genre_profiles = genre_profiles_df.values
        print(f"  -> Encoded {len(self.unique_genres)} unique genres.")

    def calculate_genre_similarity(self):
        """
        Calculates a genre-to-genre similarity matrix.
        Uses Cosine Similarity of the mean audio feature profiles for each genre.
        Resulting matrix has dimensions (num_genres x num_genres).
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

        query_genre = genre_name.strip().lower()
        
        # Perform case-insensitive match
        match = [g for g in self.unique_genres if g.lower() == query_genre]
        if not match:
            return None, f"Genre '{genre_name}' not found in the dataset."
        
        matched_genre = match[0]
        genre_idx = self.genre_to_idx[matched_genre]
        
        # Get similarities for this genre
        similarities = self.genre_similarity_matrix[genre_idx]
        
        # Sort indices descending
        sorted_indices = np.argsort(similarities)[::-1]
        
        similar_genres = []
        for idx in sorted_indices:
            if idx == genre_idx:
                continue
            
            similar_genres.append({
                "genre": self.unique_genres[idx],
                "similarity_score": similarities[idx]
            })
            
            if len(similar_genres) == top_n:
                break
                
        return pd.DataFrame(similar_genres), None

    def recommend_by_genre(self, song_name, top_n=5):
        """
        Given a song name:
        1. Finds the track in the dataset (exact or partial name matching).
        2. Retrieves the song's genre.
        3. Identifies other songs in the dataset belonging to the EXACT SAME genre.
        4. Calculates the Cosine Similarity of their audio features compared to the query song.
        5. Ranks and returns the top_n most similar tracks within that genre.
        """
        if self.df is None:
            self.load_data_and_models()

        # Find matching song (case-insensitive)
        matches = self.df[self.df["track_name"].str.lower() == song_name.lower()]

        # Partial matching fallback
        if matches.empty:
            matches = self.df[
                self.df["track_name"].str.lower().str.contains(song_name.lower(), regex=False)
            ]

        if matches.empty:
            return None, f"No song matching '{song_name}' found in the dataset."

        # Take the most popular matching track
        song_idx = self.df.index.get_loc(matches.index[0])
        song_meta = self.df.iloc[song_idx]
        
        query_genre = song_meta["track_genre"]
        
        print(f"\n[Search Match] Found song: '{song_meta['track_name']}' by {song_meta['artists']}")
        print(f"[GenreRecommender] Song genre: '{query_genre}'")
        print(f"[GenreRecommender] Filtering other tracks in genre '{query_genre}' and ranking...")

        # Filter the dataset for tracks with the same genre
        genre_df = self.df[self.df["track_genre"] == query_genre].copy()
        
        # Remove the query track from recommendations
        genre_df = genre_df[genre_df["track_id"] != song_meta["track_id"]]
        
        if genre_df.empty:
            return None, f"No other tracks found in the genre '{query_genre}' for recommendations."

        # Extract audio features for similarity calculation
        query_features = song_meta[self.scaled_feature_cols].values.reshape(1, -1)
        candidate_features = genre_df[self.scaled_feature_cols].values

        # Compute cosine similarity between the query song and candidate tracks within the genre
        similarities = cosine_similarity(query_features, candidate_features)[0]
        genre_df["similarity_score"] = similarities

        # Sort and select top_n recommendations
        rec_df = genre_df.sort_values(by="similarity_score", ascending=False).head(top_n)
        
        # Format the output DataFrame
        output_cols = ["track_name", "artists", "album_name", "track_genre", "popularity", "similarity_score"]
        rec_df = rec_df[output_cols].reset_index(drop=True)
        
        return rec_df, None


# ==============================================================================
# Package-level Wrapper Function (Required by prompt)
# ==============================================================================

_global_genre_recommender = None

def recommend_by_genre(song_name, top_n=5):
    """
    Module level helper function matching the recommend_by_genre(song_name) requirement.
    Lazily initializes the global GenreRecommender instance.
    """
    global _global_genre_recommender
    if _global_genre_recommender is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "processed", "train.csv")
        model_dir = os.path.join(base_dir, "models")
        
        _global_genre_recommender = GenreRecommender(
            cleaned_data_path=train_path,
            model_dir=model_dir
        )
    
    return _global_genre_recommender.recommend_by_genre(song_name, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("           GENRE RECOMMENDER SYSTEM TESTING BLOCK           ")
    print("=" * 60)
    
    # Instantiate recommender
    recommender = GenreRecommender()
    try:
        recommender.load_data_and_models()
        
        # Test 1: Genre similarity
        test_genre = "hardcore"
        print(f"\n--- Finding similar genres to '{test_genre}' ---")
        similar_df, err = recommender.get_similar_genres(test_genre, top_n=5)
        if err:
            print(f"Error: {err}")
        else:
            print(similar_df.to_string(index=False))

        # Test 2: Song recommendation by genre
        # Let's search for a popular song or fallback to a query
        test_song = "Blinding Lights"
        print(f"\n--- Running recommend_by_genre for '{test_song}' ---")
        rec_df, err = recommend_by_genre(test_song, top_n=5)
        if err:
            # Let's try to query the first song in train.csv instead
            first_song = recommender.df.iloc[0]["track_name"]
            print(f"Song '{test_song}' not found. Trying with first track in dataset: '{first_song}'")
            rec_df, err = recommend_by_genre(first_song, top_n=5)
            
        if err:
            print(f"Error: {err}")
        else:
            print(f"Top recommendations:")
            print(rec_df.to_string(index=False))
            
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        print("Note: Ensure that data/processed/train.csv exists by running main.py first.")
    print("=" * 60)
