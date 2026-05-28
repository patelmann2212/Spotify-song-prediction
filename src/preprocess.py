import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
    "tempo"
]

def load_data(file_path):
    """
    Loads raw CSV track data from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place your spotify_tracks.csv there.")
    
    print(f"Loading raw dataset from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_dataset(df):
    """
    Cleans the dataframe by handling missing values, duplicates, and
    scaling numeric audio features.
    """
    # 1. Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # 2. Check and drop critical missing values
    initial_shape = df.shape
    critical_cols = ["track_name", "artists"]
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    
    # 3. Handle missing values in audio features by filling with the median
    for col in AUDIO_FEATURE_COLUMNS:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            raise KeyError(f"Required audio feature column '{col}' is missing from the dataset.")
            
    # 4. Remove duplicate songs by name & artist to keep recommendations unique
    # We keep the one with the highest popularity
    if "popularity" in df.columns:
        df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
    df = df.reset_index(drop=True)
    
    cleaned_shape = df.shape
    print(f"Cleaned dataset: Removed {initial_shape[0] - cleaned_shape[0]} rows (nulls/duplicates).")
    
    # 5. Extract and scale audio features
    features = df[AUDIO_FEATURE_COLUMNS].copy()
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create a DataFrame of scaled features
    scaled_features_df = pd.DataFrame(
        scaled_features, 
        columns=[f"scaled_{col}" for col in AUDIO_FEATURE_COLUMNS]
    )
    
    # Combine original track identifiers with scaled features (keeping ONLY specified columns)
    meta_cols = [c for c in ["track_id", "track_name", "artists", "album_name", "popularity"] if c in df.columns]
    processed_df = pd.concat([df[meta_cols].reset_index(drop=True), scaled_features_df], axis=1)
    
    return processed_df, scaled_features

def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits the preprocessed dataset into train, validation, and test sets.
    Ensures validation and test sets are split evenly from the remaining (1 - train_size).
    """
    print(f"Splitting preprocessed dataset (train={train_size:.0%}, val={val_size:.0%}, test={test_size:.0%}, random_state={random_state})...")
    # First split: Separate out the training set (80%)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state
    )
    
    # Second split: Split the remaining 30% evenly (50% validation, 50% test)
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_ratio, 
        random_state=random_state
    )
    
    print(f"Dataset split completed successfully:")
    print(f"  - Train: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"  - Validation: {val_df.shape[0]} rows ({val_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"  - Test: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_processed_data(df, output_path):
    """
    Saves preprocessed and scaled tracks to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")

if __name__ == "__main__":
    # Test block for testing preprocessing pipeline and dataset splitting
    raw_path = os.path.join("data", "raw", "spotify_tracks.csv")
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "validation.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    
    if os.path.exists(raw_path):
        # 1. Load data
        df = load_data(raw_path)
        
        # 2. Preprocess data (Scaling occurs before splitting)
        processed_df, _ = preprocess_dataset(df)
        
        # 3. Split dataset (Duplicate removal has already happened during preprocessing)
        train_df, val_df, test_df = split_dataset(processed_df)
        
        # 4. Save processed datasets
        save_processed_data(train_df, train_path)
        save_processed_data(val_df, val_path)
        save_processed_data(test_df, test_path)
    else:
        print("Raw dataset not found. Run generate_dummy_data.py first to create a sample.")
