import os
import sys

# Ensure the root of the project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl / Tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    # 1. Load the processed dataset
    train_path = os.path.join("data", "processed", "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at {train_path}. Run main.py first.")
        
    print(f"Loading dataset from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Create plots directory if it does not exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set premium aesthetic styles using Seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "#1e1e24"  # Dark background color
    plt.rcParams["axes.facecolor"] = "#1e1e24"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "#b3b3b3"
    plt.rcParams["xtick.color"] = "#b3b3b3"
    plt.rcParams["ytick.color"] = "#b3b3b3"
    plt.rcParams["grid.color"] = "#2c2c35"
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    
    # Palette definition for moods
    mood_colors = {
        "Happy": "#ffd166",  # Bright yellow
        "Sad": "#118ab2",    # Melancholic blue
        "Party": "#ef476f",  # Energetic pink
        "Chill": "#06d6a0"   # Relaxed green
    }

    # ==========================================================================
    # CHART 1: Genre Distribution (Top 15 Genres)
    # ==========================================================================
    print("Generating Genre Distribution Chart...")
    plt.figure(figsize=(12, 6))
    top_genres = df["track_genre"].value_counts().head(15)
    
    # Create horizontal bar plot
    sns.barplot(
        x=top_genres.values, 
        y=top_genres.index, 
        palette="viridis", 
        hue=top_genres.index, 
        legend=False
    )
    plt.title("Top 15 Genre Distribution in Spotify Dataset", fontsize=16, fontweight="bold", color="white", pad=15)
    plt.xlabel("Number of Tracks", fontsize=12, fontweight="bold")
    plt.ylabel("Genre", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "genre_distribution.png"), dpi=150, facecolor="#1e1e24")
    plt.close()

    # ==========================================================================
    # CHART 2: Popularity Distribution
    # ==========================================================================
    print("Generating Popularity Distribution Chart...")
    plt.figure(figsize=(10, 5))
    
    # Create histogram and kernel density estimation
    sns.histplot(
        df["popularity"], 
        bins=30, 
        kde=True, 
        color="#1db954",  # Spotify green
        edgecolor="#121212"
    )
    plt.title("Track Popularity Score Distribution", fontsize=16, fontweight="bold", color="white", pad=15)
    plt.xlabel("Popularity Score (0-100)", fontsize=12, fontweight="bold")
    plt.ylabel("Count of Tracks", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "popularity_distribution.png"), dpi=150, facecolor="#1e1e24")
    plt.close()

    # ==========================================================================
    # CHART 3: Mood Distribution
    # ==========================================================================
    print("Generating Mood Distribution Chart...")
    plt.figure(figsize=(8, 5))
    
    # Make sure we have classified mood
    if "mood" in df.columns:
        mood_counts = df["mood"].value_counts()
        sns.barplot(
            x=mood_counts.index, 
            y=mood_counts.values, 
            palette=mood_colors, 
            hue=mood_counts.index, 
            legend=False
        )
        plt.title("Song Mood Distribution (Phase-2 Classifier)", fontsize=16, fontweight="bold", color="white", pad=15)
        plt.xlabel("Mood Category", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Tracks", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mood_distribution.png"), dpi=150, facecolor="#1e1e24")
    else:
        print("[Warning] 'mood' column not found in dataset. Skipping mood distribution chart.")
    plt.close()

    # ==========================================================================
    # CHART 4: Correlation Heatmap
    # ==========================================================================
    print("Generating Feature Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Define columns to correlate (using raw/scaled numeric features)
    features_to_correlate = [
        "popularity", 
        "valence", 
        "energy", 
        "scaled_danceability", 
        "scaled_loudness", 
        "scaled_speechiness", 
        "scaled_acousticness", 
        "scaled_instrumentalness", 
        "scaled_liveness", 
        "scaled_tempo"
    ]
    
    # Compute correlation matrix
    corr_matrix = df[features_to_correlate].corr()
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5, 
        annot_kws={"size": 10}, 
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    plt.title("Pearson Correlation Heatmap of Audio Features", fontsize=16, fontweight="bold", color="white", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=150, facecolor="#1e1e24")
    plt.close()

    # ==========================================================================
    # CHART 5: Top Artists Chart
    # ==========================================================================
    print("Generating Top Artists Chart...")
    plt.figure(figsize=(12, 6))
    
    # Count occurrence of artists (excluding nulls)
    top_artists = df["artists"].dropna().value_counts().head(15)
    
    # Plot horizontal bar chart
    sns.barplot(
        x=top_artists.values, 
        y=top_artists.index, 
        palette="magma", 
        hue=top_artists.index, 
        legend=False
    )
    plt.title("Top 15 Most Frequent Artists in Dataset", fontsize=16, fontweight="bold", color="white", pad=15)
    plt.xlabel("Number of Tracks", fontsize=12, fontweight="bold")
    plt.ylabel("Artist Name", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_artists.png"), dpi=150, facecolor="#1e1e24")
    plt.close()
    
    print("\nAll plots generated and saved successfully to the 'plots/' directory!")


if __name__ == "__main__":
    generate_plots()
