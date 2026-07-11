import os
import sys

# Ensure the root of the project is in python path to handle relative module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Reuse our existing recommender engines to fetch fitted representations
from src.recommend import ContentBasedRecommender
from src.genre_recommender import GenreRecommender
from src.artist_recommender import ArtistSimilarityRecommender


def load_dataset(data_path="data/processed/train.csv"):
    """
    Component 1: Dataset Loader
    Loads the preprocessed training dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}. Run main.py first.")
    return pd.read_csv(data_path)


def get_audio_similarity(query_song_row, df, scaled_feature_cols):
    """
    Component 2: Audio Similarity Calculator
    Calculates the Cosine Similarity of 10 scaled audio features
    between the query song and all songs in the dataset.
    """
    query_features = query_song_row[scaled_feature_cols].values.reshape(1, -1)
    dataset_features = df[scaled_feature_cols].values
    audio_sims = cosine_similarity(query_features, dataset_features)[0]
    return audio_sims


def get_artist_similarity(query_artist, df, artist_vectorizer):
    """
    Component 3: Artist Similarity Calculator
    Vectorizes the query artist name and all dataset artist names using TF-IDF,
    then calculates the Cosine Similarity of their TF-IDF representations.
    """
    query_vector = artist_vectorizer.transform([query_artist])
    dataset_vectors = artist_vectorizer.transform(df["artists"])
    artist_sims = cosine_similarity(query_vector, dataset_vectors)[0]
    return artist_sims


def get_genre_similarity(query_genre, df, genre_similarity_matrix, genre_to_idx, unique_genres):
    """
    Component 4: Genre Similarity Calculator
    Looks up the query genre's similarity to all other genres in the database,
    and maps the scores back to each track.
    """
    query_genre_idx = genre_to_idx[query_genre]
    similarities_for_query_genre = genre_similarity_matrix[query_genre_idx]
    
    # Create a map from genre names to similarity scores
    genre_sim_map = {
        unique_genres[i]: similarities_for_query_genre[i]
        for i in range(len(unique_genres))
    }
    # Map the scores to the tracks
    genre_sims = df["track_genre"].map(genre_sim_map).values
    return genre_sims


def get_normalized_popularity(df):
    """
    Component 5: Popularity Score Normalizer
    Applies Min-Max scaling to map raw popularity scores [0, 100] to [0, 1].
    """
    scaler = MinMaxScaler()
    normalized_pop = scaler.fit_transform(df[["popularity"]]).flatten()
    return normalized_pop


def weighted_ranking(audio_sims, artist_sims, genre_sims, pop_scores, weights=None):
    """
    Component 6: Weighted Scorer
    Combines the normalized scores using the formula:
    final_score = 0.50*audio_sim + 0.20*artist_sim + 0.20*genre_sim + 0.10*popularity
    """
    if weights is None:
        weights = {
            "audio": 0.50,
            "artist": 0.20,
            "genre": 0.20,
            "popularity": 0.10
        }
        
    final_scores = (
        weights["audio"] * audio_sims +
        weights["artist"] * artist_sims +
        weights["genre"] * genre_sims +
        weights["popularity"] * pop_scores
    )
    return final_scores


def recommend(song_name):
    """
    Final Recommendation Function:
    Assembles the Phase-2 Spotify Recommendation Pipeline:
    1. Loads data and sub-engines.
    2. Locates the query song.
    3. Runs each similarity component.
    4. Combines similarity signals with popularity using the weighted ranking formula.
    5. Sorts, filters, and returns the top 10 recommended songs formatted as requested.
    
    Returns:
    DataFrame with columns: ['Song Name', 'Artist', 'Genre', 'Popularity', 'Final Score']
    """
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    model_dir = os.path.join(base_dir, "models")
    
    # 1. Load dataset
    df = load_dataset(train_path)
    scaled_feature_cols = [col for col in df.columns if col.startswith("scaled_")]
    
    # 2. Locate the query song (case-insensitive with partial match fallback)
    matches = df[df["track_name"].str.lower() == song_name.lower()]
    if matches.empty:
        matches = df[df["track_name"].str.lower().str.contains(song_name.lower(), regex=False)]
        
    if matches.empty:
        return None, f"No song matching '{song_name}' found in the database."
        
    # Select the first matching song (highest popularity due to preprocessing)
    query_idx = df.index.get_loc(matches.index[0])
    query_song = df.iloc[query_idx]
    
    print(f"\n[Phase-2 Pipeline] Query Song: '{query_song['track_name']}' by {query_song['artists']}")
    
    # 3. Load sub-engines' precomputed structures
    # Load Genre Engine components
    genre_engine = GenreRecommender(cleaned_data_path=train_path, model_dir=model_dir)
    genre_engine.load_data_and_models()
    
    # Load Artist Engine components
    artist_engine = ArtistSimilarityRecommender(cleaned_data_path=train_path, model_dir=model_dir)
    artist_engine.load_data_and_models()

    # 4. Execute components sequentially (Pipeline flow)
    # Component A: Audio Feature Similarity
    audio_sims = get_audio_similarity(query_song, df, scaled_feature_cols)
    
    # Component B: Artist TF-IDF Similarity
    artist_sims = get_artist_similarity(query_song["artists"], df, artist_engine.vectorizer)
    
    # Component C: Genre average profile Similarity
    genre_sims = get_genre_similarity(
        query_song["track_genre"], 
        df, 
        genre_engine.genre_similarity_matrix, 
        genre_engine.genre_to_idx, 
        genre_engine.unique_genres
    )
    
    # Component D: Popularity Normalization
    pop_scores = get_normalized_popularity(df)
    
    # Component E: Weighted Score Calculation
    final_scores = weighted_ranking(audio_sims, artist_sims, genre_sims, pop_scores)

    # 5. Assemble, Sort, and Format the top 10 output
    results_df = df.copy()
    results_df["final_score"] = final_scores
    
    # Filter out the query track itself
    results_df = results_df[results_df["track_id"] != query_song["track_id"]]
    
    # Sort by final score descending and select top 10
    top_10 = results_df.sort_values(by="final_score", ascending=False).head(10)
    
    # Rename columns to match exact user request
    rename_dict = {
        "track_name": "Song Name",
        "artists": "Artist",
        "track_genre": "Genre",
        "popularity": "Popularity",
        "final_score": "Final Score"
    }
    
    # Select and format the final return columns
    top_10_formatted = top_10[list(rename_dict.keys())].rename(columns=rename_dict).reset_index(drop=True)
    
    return top_10_formatted, None


if __name__ == "__main__":
    print("=" * 60)
    print("      SPOTIFY RECOMMENDATION SYSTEM (PHASE-2 PIPELINE)      ")
    print("============================================================")
    
    test_song = "Blinding Lights"
    try:
        recommendations, err = recommend(test_song)
        if err:
            print(f"Error: {err}")
        else:
            print(f"\nTop 10 Recommendations for '{test_song}':")
            print(recommendations.to_string(index=False))
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        print("Note: Ensure data/processed/train.csv exists by running main.py first.")
    print("=" * 60)
