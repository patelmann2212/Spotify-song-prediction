import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def analyze_popularity(df):
    """
    1. ANALYZE POPULARITY DISTRIBUTION
    Analyzes the statistical distribution of the popularity column.
    """
    print("=== POPULARITY DISTRIBUTION ANALYSIS ===")
    pop_col = df["popularity"]
    mean_pop = pop_col.mean()
    median_pop = pop_col.median()
    std_pop = pop_col.std()
    min_pop = pop_col.min()
    max_pop = pop_col.max()

    percentiles = pop_col.quantile([0.25, 0.50, 0.75, 0.90])

    print(f"Mean Popularity   : {mean_pop:.2f}")
    print(f"Median Popularity : {median_pop:.2f} (50th percentile)")
    print(f"Std Deviation     : {std_pop:.2f}")
    print(f"Min Popularity    : {min_pop:.2f}")
    print(f"Max Popularity    : {max_pop:.2f}")
    print(f"25th Percentile   : {percentiles[0.25]:.2f} (Low popularity tracks)")
    print(f"75th Percentile   : {percentiles[0.75]:.2f} (Highly popular tracks)")
    print(f"90th Percentile   : {percentiles[0.90]:.2f} (Top-tier hits)")
    print("=========================================\n")


def normalize_popularity(df):
    """
    2. NORMALIZE POPULARITY SCORE
    Normalizes the popularity score column to range [0, 1] using MinMaxScaler.
    """
    print("[PopularityRanking] Normalizing popularity scores using Min-Max scaling...")
    scaler = MinMaxScaler()
    df["normalized_popularity"] = scaler.fit_transform(df[["popularity"]])

    print("  -> Created column: 'normalized_popularity'")
    print(f"  -> Normalization Range: [{df['normalized_popularity'].min():.2f}, {df['normalized_popularity'].max():.2f}]")
    print("=========================================\n")
    return df


def create_hybrid_recommendation_score(rec_df, similarity_weight=0.7, popularity_weight=0.3):
    """
    3. CREATE POPULARITY RANKING FEATURE
    Combines content-based similarity score with normalized popularity score.

    Formula:
        hybrid_score = (similarity_score * similarity_weight) + (normalized_popularity * popularity_weight)
    """
    rec_copy = rec_df.copy()

    # 1. Normalize popularity within candidate recommendations
    scaler = MinMaxScaler()
    rec_copy["norm_pop_candidate"] = scaler.fit_transform(rec_copy[["popularity"]])

    # 2. Compute weighted hybrid score
    rec_copy["hybrid_score"] = (
        (rec_copy["similarity_score"] * similarity_weight) +
        (rec_copy["norm_pop_candidate"] * popularity_weight)
    )

    # 3. Deterministic sort
    sort_cols = ["hybrid_score"]
    ascending_flags = [False]
    if "popularity" in rec_copy.columns:
        sort_cols.append("popularity")
        ascending_flags.append(False)
    if "track_id" in rec_copy.columns:
        sort_cols.append("track_id")
        ascending_flags.append(True)

    final_ranked_df = rec_copy.sort_values(by=sort_cols, ascending=ascending_flags).reset_index(drop=True)
    return final_ranked_df


if __name__ == "__main__":
    train_path = os.path.join("data", "processed", "train.csv")
    cleaned_path = os.path.join("data", "cleaned_spotify.csv")
    data_path = cleaned_path if os.path.exists(cleaned_path) else train_path

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        analyze_popularity(df)
        df = normalize_popularity(df)

        print("=== TEST HYBRID RANKING ENGINE ===")
        mock_candidates = pd.DataFrame({
            "track_name": ["Song A", "Song B", "Song C", "Song D"],
            "artists": ["Artist A", "Artist B", "Artist C", "Artist D"],
            "similarity_score": [0.98, 0.95, 0.92, 0.88],
            "popularity": [10.0, 95.0, 40.0, 90.0]
        })

        print("\nOriginal Content-Based Similarity Ranks:")
        print(mock_candidates[["track_name", "artists", "similarity_score", "popularity"]])

        hybrid_ranked = create_hybrid_recommendation_score(
            mock_candidates,
            similarity_weight=0.7,
            popularity_weight=0.3
        )

        print("\nHybrid Ranks (70% Similarity + 30% Popularity):")
        print(hybrid_ranked[["track_name", "artists", "similarity_score", "popularity", "hybrid_score"]])
        print("==================================")
    else:
        print(f"Dataset not found at {data_path}. Run preprocess.py first.")
