import os
import sys
from typing import List, Optional, Tuple, Union

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.song_lookup import find_song

# Standard continuous numerical audio features used for audio clustering
PREFERRED_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
    "tempo",
]

# Non-audio metadata columns that must never be used as clustering features
METADATA_EXCLUSION_COLUMNS = {
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "genre",
    "popularity",
    "explicit",
    "scaled_explicit",
    "mood",
    "Unnamed: 0",
}


def resolve_audio_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies available numerical audio features from the dataset.
    Prioritizes raw audio feature names, falling back to scaled_* feature names
    if raw features were omitted during preprocessing.
    Strictly excludes metadata (track_id, track_name, artists, album_name, genre, etc.).
    """
    # 1. Check for raw unscaled preferred audio features
    raw_present = [f for f in PREFERRED_AUDIO_FEATURES if f in df.columns]
    if len(raw_present) == len(PREFERRED_AUDIO_FEATURES):
        return raw_present

    # 2. Check for scaled audio feature columns (e.g. scaled_danceability)
    scaled_preferred = [f"scaled_{f}" for f in PREFERRED_AUDIO_FEATURES]
    scaled_present = [f for f in scaled_preferred if f in df.columns]
    if len(scaled_present) == len(scaled_preferred):
        return scaled_present

    # 3. Fallback: Any columns starting with 'scaled_' excluding metadata
    candidate_scaled = [
        col for col in df.columns
        if col.startswith("scaled_") and col not in METADATA_EXCLUSION_COLUMNS
    ]
    if candidate_scaled:
        return candidate_scaled

    # 4. Fallback: Any present subset of preferred features
    if raw_present:
        return raw_present

    raise ValueError(
        "No suitable numerical audio feature columns found in dataset. "
        f"Expected either {PREFERRED_AUDIO_FEATURES} or {scaled_preferred}."
    )


class KMeansRecommender:
    """
    K-Means Clustering Recommendation Engine for Spotify tracks.

    Workflow:
    1. Preprocesses numerical audio features using StandardScaler (mean=0, variance=1).
    2. Clusters songs using sklearn.cluster.KMeans into n_clusters partitions.
    3. Persists model and scaler to disk (models/kmeans_model.pkl & models/kmeans_scaler.pkl).
    4. For a query song:
       - Identifies query song's cluster assignment.
       - Isolates candidates belonging to the identical cluster.
       - Computes pairwise cosine similarity over scaled audio features.
       - Ranks intra-cluster candidates deterministically (similarity desc, popularity desc, track_id asc).
       - Returns top-N recommendations.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        n_clusters: int = 8,
        cleaned_data_path: Optional[str] = None,
        model_dir: str = "models",
        random_state: int = 42,
    ):
        if cleaned_data_path is None:
            if os.path.exists("data/cleaned_spotify.csv"):
                cleaned_data_path = "data/cleaned_spotify.csv"
            else:
                cleaned_data_path = "data/processed/train.csv"

        self.cleaned_data_path = cleaned_data_path
        self.model_dir = model_dir
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.model_path = os.path.join(model_dir, "kmeans_model.pkl")
        self.scaler_path = os.path.join(model_dir, "kmeans_scaler.pkl")

        self.df = df
        self.feature_cols: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[KMeans] = None
        self.cluster_df: Optional[pd.DataFrame] = None
        self.scaled_features: Optional[np.ndarray] = None

    def _ensure_data_loaded(self) -> None:
        """Loads dataset from cleaned_data_path if df was not supplied."""
        if self.df is None:
            if not os.path.exists(self.cleaned_data_path):
                raise FileNotFoundError(
                    f"Dataset not found at '{self.cleaned_data_path}'. Please run preprocess.py first."
                )
            self.df = pd.read_csv(self.cleaned_data_path).reset_index(drop=True)

    def fit(self, force_retrain: bool = False) -> "KMeansRecommender":
        """
        Fits or loads the K-Means clustering model and StandardScaler.
        If persisted models exist and force_retrain is False, loads them from disk.
        Otherwise, fits StandardScaler and KMeans on the audio features and saves them.
        """
        self._ensure_data_loaded()
        os.makedirs(self.model_dir, exist_ok=True)

        # Resolve numerical audio feature columns from data
        self.feature_cols = resolve_audio_feature_columns(self.df)

        models_exist = (
            os.path.exists(self.model_path)
            and os.path.exists(self.scaler_path)
        )

        if models_exist and not force_retrain:
            try:
                print(f"[KMeansRecommender] Loading saved model from {self.model_path}...")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

                # Validate feature dimension match
                raw_matrix = self._extract_clean_feature_matrix(self.df)
                if hasattr(self.scaler, "n_features_in_") and self.scaler.n_features_in_ != raw_matrix.shape[1]:
                    print("[KMeansRecommender] Feature dimension mismatch detected. Retraining...")
                    return self.fit(force_retrain=True)

                self.scaled_features = self.scaler.transform(raw_matrix)
                # Store cluster assignments in a derived DataFrame
                self.cluster_df = self.df.copy()
                self.cluster_df["cluster"] = self.model.predict(self.scaled_features)
                print(f"[KMeansRecommender] Model loaded successfully ({self.model.n_clusters} clusters).")
                return self
            except Exception as e:
                print(f"[KMeansRecommender] Failed to load saved model ({e}). Retraining...")

        # Train new model
        print(f"[KMeansRecommender] Training KMeans(k={self.n_clusters}) on {len(self.feature_cols)} audio features...")
        raw_matrix = self._extract_clean_feature_matrix(self.df)

        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(raw_matrix)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(self.scaled_features)

        # Persist model and scaler using joblib
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"[KMeansRecommender] Saved model to {self.model_path} and scaler to {self.scaler_path}.")

        # Store cluster assignments in derived DataFrame
        self.cluster_df = self.df.copy()
        self.cluster_df["cluster"] = self.model.labels_

        return self

    def _extract_clean_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts numerical audio feature matrix and handles missing/non-numeric values.
        """
        features_sub = df[self.feature_cols].copy()
        # Coerce any non-numeric values to NaN
        features_sub = features_sub.apply(pd.to_numeric, errors="coerce")
        # Impute missing values with column median
        if features_sub.isnull().any().any():
            features_sub = features_sub.fillna(features_sub.median())
        return features_sub.values.astype(float)

    def predict_cluster(self, song: Union[str, pd.Series, pd.DataFrame, np.ndarray]) -> int:
        """
        Predicts the cluster ID for a given song query, Series, or feature array.

        Parameters:
            song: Song name (str), pandas Series (song row), DataFrame, or feature vector.

        Returns:
            int: Predicted cluster index (0 to n_clusters-1).
        """
        if self.model is None or self.scaler is None:
            self.fit()

        if isinstance(song, str):
            found, err = find_song(self.df, song)
            if err:
                raise ValueError(f"Could not resolve song '{song}': {err}")
            track_id = found["track_id"]
            matched = self.df[self.df["track_id"] == track_id]
            if matched.empty:
                raise ValueError(f"Track '{track_id}' not found in dataset.")
            row = matched.iloc[0]
            raw_vec = row[self.feature_cols].values.astype(float).reshape(1, -1)
            scaled_vec = self.scaler.transform(raw_vec)
            return int(self.model.predict(scaled_vec)[0])

        elif isinstance(song, pd.Series):
            if all(col in song for col in self.feature_cols):
                raw_vec = song[self.feature_cols].values.astype(float).reshape(1, -1)
            elif "track_id" in song:
                matched = self.df[self.df["track_id"] == song["track_id"]]
                if matched.empty:
                    raise ValueError(f"Track '{song['track_id']}' not found in dataset.")
                raw_vec = matched.iloc[0][self.feature_cols].values.astype(float).reshape(1, -1)
            else:
                raise ValueError("Series does not contain required audio feature columns.")
            scaled_vec = self.scaler.transform(raw_vec)
            return int(self.model.predict(scaled_vec)[0])

        elif isinstance(song, pd.DataFrame):
            raw_matrix = song[self.feature_cols].values.astype(float)
            scaled_matrix = self.scaler.transform(raw_matrix)
            return self.model.predict(scaled_matrix)

        elif isinstance(song, np.ndarray):
            vec = song.reshape(1, -1) if song.ndim == 1 else song
            scaled_vec = self.scaler.transform(vec)
            preds = self.model.predict(scaled_vec)
            return int(preds[0]) if len(preds) == 1 else preds

        else:
            raise TypeError(f"Unsupported type for song argument: {type(song)}")

    def recommend(
        self,
        song_name: Union[str, pd.Series, dict],
        artist_name: Optional[str] = None,
        top_n: int = 10,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Recommends top_n songs within the same K-Means audio cluster as the query song.

        Workflow:
        1. Resolve song via find_song() (or reuse provided song Series/dict).
        2. Match track_id against cluster_df to extract full features and cluster.
        3. Filter candidates strictly to the same cluster.
        4. Exclude the query track itself.
        5. Compute cosine similarity across StandardScaler-normalized audio features.
        6. Rank deterministically (similarity_score desc, popularity desc, track_id asc).
        7. Return formatted DataFrame and None error, or (None, error_str).
        """
        if self.model is None or self.cluster_df is None:
            self.fit()

        # 1. Deterministic Song Lookup / Reuse already-identified song
        if isinstance(song_name, (pd.Series, dict)):
            query_song = song_name
        else:
            query_song, err = find_song(self.df, song_name, artist_name)
            if err:
                return None, err

        query_track_id = query_song["track_id"]

        # 2. Retrieve full row from cluster_df
        matched_rows = self.cluster_df[self.cluster_df["track_id"] == query_track_id]
        if matched_rows.empty:
            return None, f"Selected song '{query_song.get('track_name', song_name)}' was not found in catalog."

        query_row = matched_rows.iloc[0]
        query_cluster = int(query_row["cluster"])

        # Extract scaled query vector
        query_raw_vec = query_row[self.feature_cols].values.astype(float).reshape(1, -1)
        query_scaled_vec = self.scaler.transform(query_raw_vec)

        # 3. Filter candidates belonging strictly to the same cluster
        same_cluster_mask = self.cluster_df["cluster"] == query_cluster
        candidates_df = self.cluster_df[same_cluster_mask].copy()

        # 4. Remove the selected song by track_id
        candidates_df = candidates_df[candidates_df["track_id"] != query_track_id]
        if candidates_df.empty:
            return pd.DataFrame(), None

        # 5. Compute cosine similarity over scaled audio features
        cand_indices = candidates_df.index.values
        cand_scaled_features = self.scaled_features[cand_indices]

        similarities = cosine_similarity(query_scaled_vec, cand_scaled_features)[0]
        candidates_df["similarity_score"] = similarities

        # 6. Deterministic tie-breaking sort: similarity_score desc, popularity desc, track_id asc
        sort_cols = ["similarity_score"]
        asc_flags = [False]
        if "popularity" in candidates_df.columns:
            sort_cols.append("popularity")
            asc_flags.append(False)
        sort_cols.append("track_id")
        asc_flags.append(True)

        top_candidates = candidates_df.sort_values(by=sort_cols, ascending=asc_flags).head(top_n)

        # 7. Format result DataFrame
        output_rows = []
        for _, row in top_candidates.iterrows():
            output_rows.append(
                {
                    "track_id": row["track_id"],
                    "track_name": row["track_name"],
                    "artists": row["artists"],
                    "album_name": row.get("album_name", "N/A"),
                    "track_genre": row.get("track_genre", "N/A"),
                    "cluster_id": int(row["cluster"]),
                    "similarity_score": float(row["similarity_score"]),
                    "popularity": int(row.get("popularity", 0)),
                }
            )

        result_df = pd.DataFrame(output_rows)
        return result_df, None

    # Alias matching project convention
    get_cluster_recommendations = recommend


def compute_elbow_silhouette(
    df: Optional[pd.DataFrame] = None,
    cleaned_data_path: Optional[str] = None,
    k_range: range = range(2, 16),
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict:
    """
    Computes Elbow (inertia) and silhouette score metrics over a range of K.
    Uses a representative subsample of the dataset to keep evaluation fast.
    NOTE: This is an analytical utility and does NOT run on application startup.

    Parameters:
        df: DataFrame containing Spotify tracks.
        cleaned_data_path: CSV path if df is not loaded.
        k_range: Range of clusters to test (default K=2..15).
        sample_size: Max rows to sample for fast silhouette evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        dict: {"k_values": list, "inertias": list, "silhouette_scores": list}
    """
    from sklearn.metrics import silhouette_score

    if df is None:
        if cleaned_data_path is None:
            cleaned_data_path = (
                "data/cleaned_spotify.csv"
                if os.path.exists("data/cleaned_spotify.csv")
                else "data/processed/train.csv"
            )
        df = pd.read_csv(cleaned_data_path)

    feature_cols = resolve_audio_feature_columns(df)
    raw_matrix = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(raw_matrix)

    # Subsample for fast evaluation
    if len(scaled_matrix) > sample_size:
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(scaled_matrix), size=sample_size, replace=False)
        eval_sample = scaled_matrix[sample_indices]
    else:
        eval_sample = scaled_matrix

    k_values = list(k_range)
    inertias = []
    silhouettes = []

    print(f"[KMeans Analysis] Evaluating K from {min(k_values)} to {max(k_values)} on {len(eval_sample)} tracks...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(eval_sample)
        inertias.append(float(km.inertia_))
        try:
            sil_score = float(silhouette_score(eval_sample, labels))
        except Exception:
            sil_score = -1.0
        silhouettes.append(sil_score)
        print(f"  -> K={k:2d} | Inertia: {km.inertia_:10.2f} | Silhouette: {sil_score:.4f}")

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("           K-MEANS RECOMMENDER TESTING BLOCK                ")
    print("=" * 60)

    recommender = KMeansRecommender()
    recommender.fit()

    test_song = "Shape of You"
    print(f"\nQuerying recommendations for '{test_song}'...")
    rec_df, err = recommender.recommend(test_song, top_n=10)
    if err:
        print(f"Error: {err}")
    else:
        print(f"Found recommendations in cluster {rec_df.iloc[0]['cluster_id']}:")
        cols_to_print = ["track_name", "artists", "cluster_id", "similarity_score", "popularity"]
        print(rec_df[cols_to_print].to_string(index=False))

    print("\n" + "=" * 60)
