import os
import sys

# Ensure the root of the project is in python path to handle relative module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force terminal to use UTF-8 encoding to prevent CP1252 UnicodeEncodeErrors with international tracks
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import pandas as pd
from src.preprocess import load_data, preprocess_dataset, split_dataset, save_processed_data
from src.recommend import ContentBasedRecommender

def main():
    print("=" * 60)
    print("      SPOTIFY SONG RECOMMENDATION SYSTEM (PHASE-1)      ")
    print("             Content-Based Filtering Engine             ")
    print("=" * 60)
    
    # Define paths
    raw_dir = os.path.join("data", "raw")
    raw_path = os.path.join(raw_dir, "spotify_tracks.csv")
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "validation.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    
    # 1. Verify that raw dataset exists
    if not os.path.exists(raw_path):
        print(f"[Error] Raw dataset 'spotify_tracks.csv' not found at: {raw_path}")
        print("Please place your actual Spotify tracks dataset CSV there and re-run.")
        return
    
    # 2. Load raw data
    try:
        raw_df = load_data(raw_path)
    except Exception as e:
        print(f"[Error] Failed to load raw dataset: {e}")
        return
        
    # 3. Preprocess, normalize, and split dataset (scaling happens before splitting)
    print("\n[Step 1/3] Preprocessing raw dataset and splitting...")
    try:
        # Preprocess dataset (cleans data, handles duplicates before splitting, scales audio features)
        processed_df, _ = preprocess_dataset(raw_df)
        
        # Split processed dataset into 80% train, 10% validation, and 10% test
        train_df, val_df, test_df = split_dataset(processed_df)
        
        # Save split datasets (keeping specified meta columns and scaled features)
        save_processed_data(train_df, train_path)
        save_processed_data(val_df, val_path)
        save_processed_data(test_df, test_path)
    except Exception as e:
        print(f"[Error] Preprocessing failed: {e}")
        return
        
    # 4. Initialize and fit content-based recommendation model using training data
    print("\n[Step 2/3] Training Content-Based Recommendation Engine on Train dataset...")
    recommender = ContentBasedRecommender(train_df)
    try:
        recommender.fit()
    except Exception as e:
        print(f"[Error] Model fitting failed: {e}")
        return
        
    # Optionally load validation and test sets later for evaluation
    print("\n[Evaluation Setup] Optionally loading validation and test sets for verification:")
    try:
        if os.path.exists(val_path) and os.path.exists(test_path):
            val_loaded = pd.read_csv(val_path)
            test_loaded = pd.read_csv(test_path)
            print(f"  -> Validation set loaded successfully ({val_loaded.shape[0]} tracks, {val_loaded.shape[1]} columns)")
            print(f"  -> Test set loaded successfully ({test_loaded.shape[0]} tracks, {test_loaded.shape[1]} columns)")
    except Exception as e:
        print(f"  -> [Warning] Could not load validation/test sets: {e}")
        
    print("\n[Step 3/3] Ready for recommendations!")
    print("=" * 60)
    
    # Interactive query loop
    while True:
        # Show a few sample songs for inspiration
        print("\n--- Sample Tracks Available in Database ---")
        sample_tracks = train_df.head(8)
        for idx, row in sample_tracks.iterrows():
            print(f" • '{row['track_name']}' by {row['artists']}")
            
        print("\nOptions: ")
        print("  - Enter a song name (e.g. 'Blinding Lights' or 'Ocean Eyes')")
        print("  - Type 'exit' to quit the application")
        
        query = input("\nEnter song name for recommendations: ").strip()
        
        if query.lower() == "exit":
            print("\nThank you for using Spotify Song Recommendation System! Keep rocking!")
            print("=" * 60)
            break
            
        if not query:
            print("[Warning] Song name cannot be blank. Please try again.")
            continue
            
        # Get recommendations
        rec_df, err = recommender.get_song_recommendations(query, top_n=5)
        
        if err:
            print(f"\n[Search Result] {err}")
            print("Tip: Check spelling or try one of the sample tracks shown above.")
            continue
            
        # Display recommendations in a high-fidelity tabular view
        print("\n" + "+" + "-" * 78 + "+")
        print(f"| {'RECOMMENDED SONGS SIMILAR TO:':<76} |")
        print("+" + "-" * 78 + "+")
        print(f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Match Match':<17} |")
        print("+" + "-" * 78 + "+")
        
        for idx, row in rec_df.iterrows():
            track_str = row['track_name']
            if len(track_str) > 26:
                track_str = track_str[:23] + "..."
                
            artist_str = row['artists']
            if len(artist_str) > 20:
                artist_str = artist_str[:17] + "..."
                
            match_pct = f"{row['similarity_score'] * 100:.1f}% Match"
            
            print(f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {match_pct:<17} |")
            
        print("+" + "-" * 78 + "+")
        print("Note: Match index is calculated using mathematical Cosine Similarity of 9 audio features.")
        print("-" * 80)

if __name__ == "__main__":
    main()
