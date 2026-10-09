import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.song_lookup import find_song
from src.recommend import ContentBasedRecommender
from src.genre_recommender import GenreRecommender
from src.artist_recommender import ArtistSimilarityRecommender


def load_dataset(data_path=None):
    """
    Component 1: Dataset Loader
    Loads the recommendation catalog or training dataset.
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cleaned_path = os.path.join(base_dir, "data", "cleaned_spotify.csv")
        train_path = os.path.join(base_dir, "data", "processed", "train.csv")
        data_path = cleaned_path if os.path.exists(cleaned_path) else train_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocess.py or main.py first.")
    return pd.read_csv(data_path).reset_index(drop=True)


def get_audio_similarity(query_song_row, df, scaled_feature_cols):
    """
    Component 2: Audio Similarity Calculator
    Calculates Cosine Similarity of scaled audio features
    between query song and all songs in the dataset.
    """
    query_features = query_song_row[scaled_feature_cols].values.astype(float).reshape(1, -1)
    dataset_features = df[scaled_feature_cols].values.astype(float)
    return cosine_similarity(query_features, dataset_features)[0]


def get_artist_similarity(query_artist, df, artist_vectorizer):
    """
    Component 3: Artist Similarity Calculator
    Vectorizes query artist name and dataset artists using TF-IDF,
    calculating Cosine Similarity of token representations.
    """
    query_vector = artist_vectorizer.transform([str(query_artist)])
    dataset_vectors = artist_vectorizer.transform(df["artists"].astype(str))
    return cosine_similarity(query_vector, dataset_vectors)[0]


def get_genre_similarity(query_genre, df, genre_similarity_matrix, genre_to_idx, unique_genres):
    """
    Component 4: Genre Similarity Calculator
    Maps query genre's profile similarity to all tracks in dataset.
    """
    query_genre_str = str(query_genre)
    if query_genre_str in genre_to_idx:
        query_genre_idx = genre_to_idx[query_genre_str]
        similarities_for_query_genre = genre_similarity_matrix[query_genre_idx]
        genre_sim_map = {
            unique_genres[i]: similarities_for_query_genre[i]
            for i in range(len(unique_genres))
        }
        return df["track_genre"].map(genre_sim_map).fillna(0.0).values
    else:
        return np.zeros(len(df))


def get_normalized_popularity(df):
    """
    Component 5: Popularity Score Normalizer
    Applies Min-Max scaling to map raw popularity scores to [0, 1].
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(df[["popularity"]]).flatten()


def weighted_ranking(audio_sims, artist_sims, genre_sims, pop_scores, weights=None):
    """
    Component 6: Weighted Scorer
    Combines normalized scores using:
    final_score = 0.50*audio_sim + 0.20*artist_sim + 0.20*genre_sim + 0.10*popularity
    """
    if weights is None:
        weights = {
            "audio": 0.50,
            "artist": 0.20,
            "genre": 0.20,
            "popularity": 0.10
        }

    return (
        weights["audio"] * audio_sims +
        weights["artist"] * artist_sims +
        weights["genre"] * genre_sims +
        weights["popularity"] * pop_scores
    )


def recommend(song_name, artist_name=None, top_n=10):
    """
    Final Recommendation Function (Phase-2 Spotify Pipeline):
    1. Loads dataset and sub-engines.
    2. Deterministically locates the query song (and optional artist).
    3. Runs each similarity component.
    4. Combines similarity signals with popularity using the weighted ranking formula.
    5. Sorts deterministically, filters query track, and returns top recommendations.

    Returns:
    DataFrame with columns: ['Song Name', 'Artist', 'Genre', 'Popularity', 'Final Score']
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, "data", "cleaned_spotify.csv")
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    dataset_path = cleaned_path if os.path.exists(cleaned_path) else train_path
    model_dir = os.path.join(base_dir, "models")

    # 1. Load dataset
    df = load_dataset(dataset_path)
    scaled_feature_cols = [col for col in df.columns if col.startswith("scaled_")]

    # 2. Deterministic track lookup
    query_song, err = find_song(df, song_name, artist_name)
    if err:
        return None, err

    print(f"\n[Phase-2 Pipeline] Query Song: '{query_song['track_name']}' by {query_song['artists']} (Popularity: {query_song.get('popularity', 'N/A')})")

    # 3. Load sub-engines
    genre_engine = GenreRecommender(cleaned_data_path=dataset_path, model_dir=model_dir)
    genre_engine.load_data_and_models()

    artist_engine = ArtistSimilarityRecommender(cleaned_data_path=dataset_path, model_dir=model_dir)
    artist_engine.load_data_and_models()

    # 4. Execute components
    audio_sims = get_audio_similarity(query_song, df, scaled_feature_cols)
    artist_sims = get_artist_similarity(query_song["artists"], df, artist_engine.vectorizer)
    genre_sims = get_genre_similarity(
        query_song.get("track_genre", ""),
        df,
        genre_engine.genre_similarity_matrix,
        genre_engine.genre_to_idx,
        genre_engine.unique_genres
    )
    pop_scores = get_normalized_popularity(df)
    final_scores = weighted_ranking(audio_sims, artist_sims, genre_sims, pop_scores)

    # 5. Assemble, Sort, and Format top output
    results_df = df.copy()
    results_df["final_score"] = final_scores

    # Filter out query track by track_id
    results_df = results_df[results_df["track_id"] != query_song["track_id"]]

    # Sort deterministically
    top_candidates = results_df.sort_values(
        by=["final_score", "popularity", "track_id"],
        ascending=[False, False, True]
    ).head(top_n)

    rename_dict = {
        "track_name": "Song Name",
        "artists": "Artist",
        "track_genre": "Genre",
        "popularity": "Popularity",
        "final_score": "Final Score"
    }

    cols = [c for c in list(rename_dict.keys()) if c in top_candidates.columns]
    top_formatted = top_candidates[cols].rename(columns=rename_dict).reset_index(drop=True)

    return top_formatted, None


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
    print("=" * 60)
