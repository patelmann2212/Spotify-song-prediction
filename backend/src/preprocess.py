import os
import sys

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.song_lookup import normalize_text, extract_base_title

# Define the standard list of audio features used for similarity
AUDIO_FEATURE_COLUMNS = [
    "explicit",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_data(file_path):
    """
    Loads raw CSV track data from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please place your spotify_tracks.csv there."
        )

    print(f"Loading raw dataset from {file_path}...")
    return pd.read_csv(file_path)


def clean_dataset(df):
    """
    Cleans raw track data by stripping column whitespace, dropping critical missing
    values, imputing missing audio features, and deduplicating identical tracks.

    Deduplication keeps the highest popularity entry for identical (track_name, artists) pairs,
    while preserving covers/different artist versions as separate searchable tracks.
    """
    df = df.copy()

    # 1. Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # 2. Check and drop critical missing values
    initial_shape = df.shape
    critical_cols = ["track_name", "artists"]
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns]).copy()

    # 3. Handle missing values in audio features by filling with median
    for col in AUDIO_FEATURE_COLUMNS:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            raise KeyError(
                f"Required audio feature column '{col}' is missing from the dataset."
            )

    # 4. Remove duplicate songs by name & artist, keeping the one with highest popularity
    if "popularity" in df.columns:
        df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
    df = df.reset_index(drop=True)

    cleaned_shape = df.shape
    print(
        f"Cleaned dataset: Removed {initial_shape[0] - cleaned_shape[0]} rows (nulls/duplicates)."
    )
    return df


def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits the unscaled, cleaned dataset into train, validation, and test sets using
    a group-aware split strategy.

    Group Key:
      base_title (normalized title without version tags) + '___' + normalized_artists

    Why Group-Aware Split?
      Standard random splitting risks placing remixes, live performances, or acoustic versions
      of the same song by the same artist across both train and test splits, causing data leakage.
      GroupShuffleSplit ensures all variations of a logical track remain together in either
      train, validation, or test.
    """
    print(
        f"Splitting dataset with group-aware strategy (train={train_size:.0%}, val={val_size:.0%}, test={test_size:.0%}, random_state={random_state})..."
    )

    # Construct grouping key
    if "track_name" in df.columns and "artists" in df.columns:
        group_key = (
            df["track_name"].apply(extract_base_title)
            + "___"
            + df["artists"].apply(normalize_text)
        )
    else:
        group_key = None

    if group_key is not None and group_key.nunique() > 1:
        # First split: Train vs (Validation + Test)
        gss_train = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, temp_idx = next(gss_train.split(df, groups=group_key))
        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()
        temp_group_key = group_key.iloc[temp_idx]

        # Second split: Validation vs Test from remaining temp split
        val_ratio = val_size / (val_size + test_size)
        gss_val = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=random_state)
        val_idx_rel, test_idx_rel = next(gss_val.split(temp_df, groups=temp_group_key))
        val_df = temp_df.iloc[val_idx_rel].copy()
        test_df = temp_df.iloc[test_idx_rel].copy()
    else:
        # Fallback to standard train_test_split if grouping key is unavailable
        train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=random_state)

    # Reset indices to ensure clean consecutive integer indexing
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Dataset group-aware split completed successfully:")
    print(f"  - Train: {train_df.shape[0]} rows ({train_df.shape[0] / df.shape[0] * 100:.1f}%)")
    print(f"  - Validation: {val_df.shape[0]} rows ({val_df.shape[0] / df.shape[0] * 100:.1f}%)")
    print(f"  - Test: {test_df.shape[0]} rows ({test_df.shape[0] / df.shape[0] * 100:.1f}%)")

    return train_df, val_df, test_df


def fit_scaler(train_df):
    """
    Fits MinMaxScaler strictly on the TRAINING set audio features.
    Prevents any data leakage from validation or test sets.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[AUDIO_FEATURE_COLUMNS])
    return scaler


def classify_mood_rule_based(scaled_energy, scaled_tempo, scaled_valence):
    """
    Deterministic rule-based mood classification:
    - Party: Fast tempo (>0.55) & High energy (>0.65)
    - Chill: Slow tempo (<0.45) & Quiet energy (<0.40)
    - Happy: High positivity/valence (>0.50)
    - Sad: Low positivity/valence & lower energy
    """
    if scaled_energy > 0.65 and scaled_tempo > 0.55:
        return "Party"
    elif scaled_energy < 0.40 and scaled_tempo < 0.45:
        return "Chill"
    elif scaled_valence > 0.50:
        return "Happy"
    else:
        return "Sad"


def transform_dataset_features(df, scaler):
    """
    Transforms audio features using the pre-fitted training scaler and builds
    the final processed DataFrame containing metadata and scaled features.
    """
    scaled_features = scaler.transform(df[AUDIO_FEATURE_COLUMNS])
    scaled_feature_cols = [f"scaled_{col}" for col in AUDIO_FEATURE_COLUMNS]
    scaled_df = pd.DataFrame(scaled_features, columns=scaled_feature_cols, index=df.index)

    meta_cols = [
        c for c in [
            "track_id", "track_name", "artists", "album_name",
            "popularity", "track_genre", "valence", "energy"
        ] if c in df.columns
    ]

    processed = pd.concat([df[meta_cols], scaled_df], axis=1).reset_index(drop=True)

    # Attach rule-based mood column using scaled features
    processed["mood"] = [
        classify_mood_rule_based(e, t, v)
        for e, t, v in zip(
            processed["scaled_energy"],
            processed["scaled_tempo"],
            processed["scaled_valence"]
        )
    ]

    return processed


def preprocess_pipeline(raw_df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Full end-to-end preprocessing pipeline without data leakage:
    1. Basic Cleaning & Deduplication of raw data.
    2. Group-aware Train / Validation / Test split.
    3. Fit MinMaxScaler ONLY on Training set.
    4. Transform Training set with training scaler.
    5. Transform Validation set with training scaler.
    6. Transform Test set with training scaler.
    7. Transform full deduplicated catalog with training scaler for search/recommendations.

    Returns:
        catalog_df, train_df, val_df, test_df, scaler
    """
    # 1. Clean & deduplicate
    cleaned_df = clean_dataset(raw_df)

    # 2. Split (group-aware) BEFORE scaling
    train_unscaled, val_unscaled, test_unscaled = split_dataset(
        cleaned_df, train_size=train_size, val_size=val_size, test_size=test_size, random_state=random_state
    )

    # 3. Fit scaler ONLY on training data
    print("Fitting MinMaxScaler ONLY on the training dataset (zero leakage)...")
    scaler = fit_scaler(train_unscaled)

    # 4. Transform all sets using the training scaler
    print("Transforming training, validation, test, and catalog datasets using training scaler...")
    train_processed = transform_dataset_features(train_unscaled, scaler)
    val_processed = transform_dataset_features(val_unscaled, scaler)
    test_processed = transform_dataset_features(test_unscaled, scaler)
    catalog_processed = transform_dataset_features(cleaned_df, scaler)

    return catalog_processed, train_processed, val_processed, test_processed, scaler


def preprocess_dataset(df):
    """
    Compatibility wrapper matching legacy function signature:
    processed_df, scaled_features, scaler = preprocess_dataset(df)
    """
    catalog_df, train_df, _, _, scaler = preprocess_pipeline(df)
    scaled_feature_cols = [f"scaled_{col}" for col in AUDIO_FEATURE_COLUMNS]
    scaled_features = catalog_df[scaled_feature_cols].values
    return catalog_df, scaled_features, scaler


def save_processed_data(df, output_path):
    """
    Saves preprocessed tracks to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")


def save_scaler(scaler, output_path):
    """
    Saves the fitted scaler object to disk using pickle.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {output_path}")


if __name__ == "__main__":
    raw_path_root = os.path.join("data", "spotify_tracks.csv")
    raw_path_raw = os.path.join("data", "raw", "spotify_tracks.csv")

    if os.path.exists(raw_path_root):
        raw_path = raw_path_root
    else:
        raw_path = raw_path_raw

    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "validation.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    cleaned_path = os.path.join("data", "cleaned_spotify.csv")

    if os.path.exists(raw_path):
        # 1. Load data
        raw_df = load_data(raw_path)

        # 2. Run complete leak-free preprocessing pipeline
        catalog_df, train_df, val_df, test_df, scaler = preprocess_pipeline(raw_df)

        # 3. Save searchable recommendation catalog to data/cleaned_spotify.csv
        save_processed_data(catalog_df, cleaned_path)

        # 4. Save evaluation split datasets to data/processed/
        save_processed_data(train_df, train_path)
        save_processed_data(val_df, val_path)
        save_processed_data(test_df, test_path)

        # 5. Save fitted scaler to models/scaler.pkl
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
        save_scaler(scaler, scaler_path)
    else:
        print(
            f"Raw dataset not found at {raw_path_root} or {raw_path_raw}. Please ensure you have spotify_tracks.csv."
        )
