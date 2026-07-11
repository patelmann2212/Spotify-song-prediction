import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def analyze_popularity(df):
    """
    1. ANALYZE POPULARITY DISTRIBUTION
    This function analyzes the statistical distribution of the popularity column.
    Understanding the distribution is crucial before deciding on normalization.
    """
    print("=== POPULARITY DISTRIBUTION ANALYSIS ===")
    
    # Calculate key descriptive statistics
    pop_col = df["popularity"]
    mean_pop = pop_col.mean()
    median_pop = pop_col.median()
    std_pop = pop_col.std()
    min_pop = pop_col.min()
    max_pop = pop_col.max()
    
    # Calculate percentiles (25%, 50%, 75%, 90%)
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
    Normalizes the popularity score column to a range of [0, 1].
    
    Why Normalize?
    Audio similarity scores (like Cosine Similarity) are in the range of [-1, 1] or [0, 1].
    To blend popularity with similarity scores, they MUST be on the same scale [0, 1].
    Otherwise, popularity (0-100) would completely overpower similarity (0-1).
    """
    print("[PopularityRanking] Normalizing popularity scores using Min-Max scaling...")
    
    # We use scikit-learn's MinMaxScaler to scale the feature to [0, 1] range
    # Formula: scaled = (x - min) / (max - min)
    scaler = MinMaxScaler()
    
    # Fit and transform the popularity column
    # Reshaping is required by scikit-learn (.values.reshape(-1, 1))
    df["normalized_popularity"] = scaler.fit_transform(df[["popularity"]])
    
    print("  -> Created column: 'normalized_popularity'")
    print(f"  -> Normalization Range: [{df['normalized_popularity'].min():.2f}, {df['normalized_popularity'].max():.2f}]")
    print("=========================================\n")
    return df


def create_hybrid_recommendation_score(rec_df, similarity_weight=0.7, popularity_weight=0.3):
    """
    3. CREATE POPULARITY RANKING FEATURE
    Computes a hybrid recommendation score by combining the content-based similarity score
    with the normalized popularity score.
    
    Formula:
        hybrid_score = (similarity_score * similarity_weight) + (normalized_popularity * popularity_weight)
        
    Parameters:
    - rec_df: DataFrame of initial content recommendations (must have similarity_score and popularity columns)
    - similarity_weight: Influence of audio feature similarity (e.g. 0.70)
    - popularity_weight: Influence of general popularity (e.g. 0.30)
    """
    # 1. Normalize popularity within the candidate recommendations
    scaler = MinMaxScaler()
    rec_df["norm_pop_candidate"] = scaler.fit_transform(rec_df[["popularity"]])
    
    # 2. Compute the weighted hybrid score
    rec_df["hybrid_score"] = (
        (rec_df["similarity_score"] * similarity_weight) + 
        (rec_df["norm_pop_candidate"] * popularity_weight)
    )
    
    # 3. Sort the candidates by the hybrid score descending
    final_ranked_df = rec_df.sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)
    
    return final_ranked_df


# ==============================================================================
# Why Popularity Can Improve Recommendations (Explanation)
# ==============================================================================
"""
EXPLANATION: WHY POPULARITY IMPROVES RECOMMENDATIONS

1. Mitigating Content-Based Obscurity Bias:
   Pure content-based similarity might recommend a song with 99% similarity that has 
   0 popularity (e.g. amateur bedroom recordings or system noise tracks). While mathematically 
   correct, this often results in a poor user experience. Introducing popularity bias ensures 
   recommending songs that have achieved some degree of validation/quality check from the public.

2. Mimicking Social Proof & Broad Appeal:
   Humans are social listeners. Users are significantly more likely to engage with and enjoy 
   recommendations if they contain a mixture of songs they already recognize or songs that 
   other users are active listening to. Popularity acts as a proxy for "social proof".

3. Handling the Cold-Start Problem (Fallback):
   When we lack historical listening data for a new user, recommending popular songs (either 
   globally or within a specific genre/mood) is the safest baseline starting point (fallback).

4. Finding the Sweet Spot (Personalization vs. Trends):
   By combining Cosine Similarity (personalization) with Popularity (collective trends) 
   using a weighted average:
       Score = 70% Similarity + 30% Popularity
   We get the best of both worlds: highly tailored song recommendations that still feel 
   familiar and engaging.
"""

if __name__ == "__main__":
    # Test script block to verify implementation details
    train_path = os.path.join("data", "processed", "train.csv")
    
    if os.path.exists(train_path):
        # Load processed training data
        df = pd.read_csv(train_path)
        
        # 1. Analyze popularity distribution
        analyze_popularity(df)
        
        # 2. Normalize popularity scores
        df = normalize_popularity(df)
        
        # 3. Test the hybrid score on a mock candidate dataset
        print("=== TEST HYBRID RANKING ENGINE ===")
        # Creating a mock candidate list of similar songs
        mock_candidates = pd.DataFrame({
            "track_name": ["Song A", "Song B", "Song C", "Song D"],
            "artists": ["Artist A", "Artist B", "Artist C", "Artist D"],
            "similarity_score": [0.98, 0.95, 0.92, 0.88],  # Obscure song matches closely
            "popularity": [10.0, 95.0, 40.0, 90.0]         # Song B and D are massive hits
        })
        
        print("\nOriginal Content-Based Similarity Ranks (Sorted by Similarity):")
        print(mock_candidates[["track_name", "artists", "similarity_score", "popularity"]])
        
        # Apply hybrid ranking
        hybrid_ranked = create_hybrid_recommendation_score(
            mock_candidates, 
            similarity_weight=0.7, 
            popularity_weight=0.3
        )
        
        print("\nHybrid Ranks (70% Similarity + 30% Popularity):")
        print(hybrid_ranked[["track_name", "artists", "similarity_score", "popularity", "hybrid_score"]])
        print("\nNote: 'Song B' (95% similarity, 95 popularity) jumped to Rank 1 over 'Song A' (98% similarity, 10 popularity).")
        print("==================================")
    else:
        print(f"Training dataset not found at {train_path}. Run main.py first.")
