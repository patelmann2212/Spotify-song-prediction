import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ArtistSimilarityRecommender:
    """
    Artist Similarity Recommender module for Spotify Recommendation System.
    Uses lazy loading to retrieve artist data from the existing data/processed/train.csv only when needed.
    Learns name token distributions using TF-IDF directly on the unique artist strings (e.g. collaborations),
    and calculates similarity scores on-the-fly using Cosine Similarity.
    """
    def __init__(self, cleaned_data_path="data/processed/train.csv", model_dir="models"):
        # Save file paths for lazy loading
        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "artist_similarity.pkl")
        
        # Placeholders for lazy loading
        self.unique_artists = None       # Pandas Series of unique artist strings
        self.vectorizer = None           # Fitted TfidfVectorizer
        self.tfidf_matrix = None         # Sparse TF-IDF feature matrix of unique artists
        self.artist_to_idx = None        # Map from lowercas"e artist name to matrix row index

    def load_data_and_models(self):
        """
        Lazy-loads unique artists and the TF-IDF representation.
        If a precomputed state exists in models/artist_similarity.pkl, loads it to save time.
        Otherwise, loads the 'artists' column of train.csv, fits a TF-IDF vectorizer on the
        unique artist names, and saves the resulting model state.
        """
        # Ensure the output models directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # 1. Attempt to load the pre-saved state to skip rebuilding
        if os.path.exists(self.model_path):
            print(f"[Lazy Load] Loading precomputed artist similarity state from {self.model_path}...")
            try:
                with open(self.model_path, "rb") as f:
                    saved_model = pickle.load(f)
                    self.unique_artists = saved_model["unique_artists"]
                    self.vectorizer = saved_model["vectorizer"]
                    self.tfidf_matrix = saved_model["tfidf_matrix"]
                    self.artist_to_idx = saved_model["artist_to_idx"]
                print("[Lazy Load] Model loaded successfully.")
                return
            except Exception as e:
                print(f"[Warning] Failed to load cached model: {e}. Rebuilding from train.csv...")

        # 2. If no saved model, lazy load ONLY the 'artists' column from the existing train.csv
        print(f"[Lazy Load] Loading artist column from {self.cleaned_data_path}...")
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Training dataset not found at {self.cleaned_data_path}. Please run Phase-1 preprocessing first."
            )
        
        # Read only the artists column to keep memory footprint minimal
        df = pd.read_csv(self.cleaned_data_path, usecols=["artists"])
        
        # Clean null values and strip whitespace
        df = df.dropna(subset=["artists"])
        df["artists"] = df["artists"].str.strip()

        # 3. Extract unique artist strings (retains collaborations like "The Weeknd; Daft Punk" as a single entry)
        print("Extracting unique artist strings from training set...")
        unique_list = df["artists"].unique()
        self.unique_artists = pd.Series(unique_list, name="artists")

        # 4. Apply TF-IDF Vectorizer to unique artist names
        # Token pattern keeps full words/names as separate tokens (lowercased by default)
        print("Applying TF-IDF Vectorizer to unique artist names...")
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.unique_artists)

        # 5. Build lookup dictionary for O(1) index searches
        self.artist_to_idx = {
            name.lower(): idx for idx, name in enumerate(self.unique_artists)
        }

        # 6. Save the trained state to disk (small sparse matrix representation)
        self.save_model()

    def save_model(self):
        """
        Saves the fitted model state to disk using pickle.
        """
        model_state = {
            "unique_artists": self.unique_artists,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "artist_to_idx": self.artist_to_idx,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(model_state, f)
        print(f"Artist similarity model saved to {self.model_path} ({len(self.unique_artists)} unique artists)")

    def get_similar_artists(self, artist_name, top_n=10):
        """
        Returns the top N most similar artists to the query artist_name based on TF-IDF name overlap.
        Computes cosine similarity on-the-fly for rapid query execution and low memory usage.
        """
        # Ensure that unique artist lists and vectorizers are loaded
        if self.tfidf_matrix is None:
            self.load_data_and_models()

        query_clean = artist_name.strip().lower()

        # 1. Look up the index of the query artist (exact match first)
        if query_clean in self.artist_to_idx:
            artist_idx = self.artist_to_idx[query_clean]
        else:
            # Fallback to partial matches if name is written slightly differently
            matches = [name for name in self.artist_to_idx if query_clean in name]
            if not matches:
                return None, f"No artist matching '{artist_name}' found in the dataset."
            
            # Select first partial match
            matched_artist = matches[0]
            artist_idx = self.artist_to_idx[matched_artist]
            print(f"[Search Match] Exact match for '{artist_name}' not found. Using partial match: '{self.unique_artists.iloc[artist_idx]}'")

        # 2. Extract the query artist's TF-IDF sparse vector (shape: 1 x V)
        query_vector = self.tfidf_matrix[artist_idx]

        # 3. Calculate Cosine Similarity on-the-fly against all unique artist vectors (shape: 1 x N)
        # We flatten the matrix output [0] to a 1D numpy array of size N
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # 4. Sort indices in descending order of similarity score (highest similarity first)
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # 5. Retrieve top N recommendations, skipping the query artist itself
        recommendations = []
        for idx in sorted_indices:
            # Skip the query artist
            if idx == artist_idx:
                continue

            rec_artist = self.unique_artists.iloc[idx]
            score = similarity_scores[idx]

            # Since we are vectorizing name tokens, a similarity score of 0 indicates no name overlap
            if score == 0.0:
                continue

            recommendations.append({
                "artist": rec_artist,
                "similarity_score": score
            })

            # Break once we've collected the required number of similar artists
            if len(recommendations) == top_n:
                break

        # Return recommendation list as a pandas DataFrame
        return pd.DataFrame(recommendations), None


# ==============================================================================
# Package-level Wrapper Function
# ==============================================================================

_global_recommender = None

def get_similar_artists(artist_name, top_n=10):
    """
    Module level helper function matching the get_similar_artists(artist_name) requirement.
    Lazily initializes the similarity recommender instance.
    """
    global _global_recommender
    if _global_recommender is None:
        # Determine paths relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "processed", "train.csv")
        model_dir = os.path.join(base_dir, "models")
        
        _global_recommender = ArtistSimilarityRecommender(
            cleaned_data_path=train_path, 
            model_dir=model_dir
        )
    
    return _global_recommender.get_similar_artists(artist_name, top_n)


if __name__ == "__main__":
    # Test script block to verify the implementation independently
    print("=" * 60)
    print("           ARTIST SIMILARITY TESTING BLOCK            ")
    print("=" * 60)
    
    # Run the recommender with a test name from train.csv
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
        print("Note: Make sure data/processed/train.csv exists by running main.py or preprocess.py first.")
    print("=" * 60)
