import os
import sys

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots():
    # 1. Load dataset (check train.csv first, then cleaned_spotify.csv)
    train_path = os.path.join("data", "processed", "train.csv")
    catalog_path = os.path.join("data", "cleaned_spotify.csv")
    data_path = train_path if os.path.exists(train_path) else catalog_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocess.py first.")

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Create plots directory if it does not exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Set styles using Seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "#1e1e24"
    plt.rcParams["axes.facecolor"] = "#1e1e24"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "#b3b3b3"
    plt.rcParams["xtick.color"] = "#b3b3b3"
    plt.rcParams["ytick.color"] = "#b3b3b3"
    plt.rcParams["grid.color"] = "#2c2c35"
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    mood_colors = {
        "Happy": "#ffd166",
        "Sad": "#118ab2",
        "Party": "#ef476f",
        "Chill": "#06d6a0"
    }

    # ==========================================================================
    # CHART 1: Genre Distribution (Top 15 Genres)
    # ==========================================================================
    print("Generating Genre Distribution Chart...")
    plt.figure(figsize=(12, 6))
    top_genres = df["track_genre"].value_counts().head(15)

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

    sns.histplot(
        df["popularity"],
        bins=30,
        kde=True,
        color="#1db954",
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

    if "mood" in df.columns:
        mood_counts = df["mood"].value_counts()
        sns.barplot(
            x=mood_counts.index,
            y=mood_counts.values,
            palette=mood_colors,
            hue=mood_counts.index,
            legend=False
        )
        plt.title("Song Mood Distribution (Rule-Based Classifier)", fontsize=16, fontweight="bold", color="white", pad=15)
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

    candidate_features = [
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
    features_to_correlate = [f for f in candidate_features if f in df.columns]

    corr_matrix = df[features_to_correlate].corr()

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

    top_artists = df["artists"].dropna().value_counts().head(15)

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

    # ==========================================================================
    # CHART 6: K-Means Audio Clusters (2D PCA Visualization)
    # ==========================================================================
    try:
        plot_kmeans_clusters(df=df, plots_dir=plots_dir)
    except Exception as e:
        print(f"[Warning] Could not generate K-Means PCA plot: {e}")

    print("\nAll plots generated and saved successfully to the 'plots/' directory!")


def plot_kmeans_clusters(
    df=None,
    model_dir="models",
    plots_dir="plots",
    sample_size=3000,
    random_state=42
):
    """
    Generates a 2D PCA projection scatter plot of audio features colored by K-Means cluster.
    PCA is strictly used for 2D visualization and dimensionality reduction.
    Does NOT affect or run during standard recommendation application execution.
    """
    from sklearn.decomposition import PCA
    from src.kmeans_recommender import KMeansRecommender, resolve_audio_feature_columns

    os.makedirs(plots_dir, exist_ok=True)

    if df is None:
        catalog_path = "data/cleaned_spotify.csv"
        train_path = "data/processed/train.csv"
        data_path = catalog_path if os.path.exists(catalog_path) else train_path
        df = pd.read_csv(data_path)

    recommender = KMeansRecommender(df=df, model_dir=model_dir, random_state=random_state)
    recommender.fit()

    feature_cols = resolve_audio_feature_columns(df)
    clean_features = recommender.scaled_features
    cluster_labels = recommender.cluster_df["cluster"].values

    # Subsample for a clear, fast, and uncrowded scatter plot
    total_samples = len(clean_features)
    if total_samples > sample_size:
        import numpy as np
        np.random.seed(random_state)
        idx_sample = np.random.choice(total_samples, size=sample_size, replace=False)
        features_sub = clean_features[idx_sample]
        labels_sub = cluster_labels[idx_sample]
    else:
        features_sub = clean_features
        labels_sub = cluster_labels

    print("Generating 2D PCA Projection of K-Means Audio Clusters...")
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(features_sub)

    var_explained = pca.explained_variance_ratio_ * 100
    print(f"  -> PCA Explained Variance: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")

    plt.figure(figsize=(10, 7))
    pca_df = pd.DataFrame({
        "PC1": coords_2d[:, 0],
        "PC2": coords_2d[:, 1],
        "Cluster": [f"Cluster {c}" for c in labels_sub]
    })

    unique_clusters = sorted(pca_df["Cluster"].unique(), key=lambda x: int(x.split()[1]))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        hue_order=unique_clusters,
        palette="tab10",
        alpha=0.75,
        s=30,
        edgecolor="none"
    )

    plt.title(
        f"K-Means Audio Clusters ({recommender.n_clusters} Clusters via 2D PCA Projection)",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=15
    )
    plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1f}% Variance)", fontsize=12, fontweight="bold")
    plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1f}% Variance)", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, facecolor="#1e1e24", edgecolor="#2c2c35")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "kmeans_clusters_pca.png")
    plt.savefig(out_path, dpi=150, facecolor="#1e1e24")
    plt.close()
    print(f"  -> Saved K-Means PCA plot to {out_path}")


if __name__ == "__main__":
    generate_plots()
