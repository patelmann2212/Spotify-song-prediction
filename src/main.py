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
from src.preprocess import (
    load_data,
    preprocess_dataset,
    split_dataset,
    save_processed_data,
    save_scaler,
)
from src.recommend import ContentBasedRecommender
from src.genre_recommender import GenreRecommender
from src.artist_recommender import ArtistSimilarityRecommender
from src.mood_recommender import MoodRecommender
from src.weighted_recommender import WeightedRecommender


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
    models_dir = os.path.join("models")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    similarity_matrix_path = os.path.join(models_dir, "similarity.pkl")

    # 1. Use processed data if available, otherwise preprocess raw dataset
    if os.path.exists(train_path):
        print(f"[Info] Found processed dataset at {train_path}. Loading train set...")
        try:
            train_df = pd.read_csv(train_path)
            print(
                f"  -> Loaded train set ({train_df.shape[0]} tracks, {train_df.shape[1]} columns)"
            )
        except Exception as e:
            print(f"[Error] Failed to load processed train dataset: {e}")
            return
    else:
        # If processed train set doesn't exist, ensure raw dataset is present
        if not os.path.exists(raw_path):
            print(f"[Error] Raw dataset 'spotify_tracks.csv' not found at: {raw_path}")
            print(
                "Please place your actual Spotify tracks dataset CSV there and re-run."
            )
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
            processed_df, _, scaler = preprocess_dataset(raw_df)

            # Split processed dataset into 80% train, 10% validation, and 10% test
            train_df, val_df, test_df = split_dataset(processed_df)

            # Save split datasets (keeping specified meta columns and scaled features)
            save_processed_data(train_df, train_path)
            save_processed_data(val_df, val_path)
            save_processed_data(test_df, test_path)

            # Save the fitted scaler for later inference or transformation
            save_scaler(scaler, scaler_path)
            print(
                f"  -> Preprocessing complete. Saved processed files to data/processed/ and scaler to {scaler_path}"
            )
        except Exception as e:
            print(f"[Error] Preprocessing failed: {e}")
            return

    # 4. Initialize and fit content-based recommendation model using training data
    print(
        "\n[Step 2/3] Training Content-Based Recommendation Engine on Train dataset..."
    )
    recommender = ContentBasedRecommender(train_df)
    try:
        recommender.fit(build_matrix=True)
        recommender.save_similarity_matrix(similarity_matrix_path)
    except Exception as e:
        print(f"[Error] Model fitting failed: {e}")
        return

    # 4.5. Initialize and fit Genre Recommendation Engine
    print(
        "\n[Step 2.5/3] Preparing Genre Recommendation Engine..."
    )
    genre_recommender = GenreRecommender(cleaned_data_path=train_path, model_dir=models_dir)
    try:
        genre_recommender.load_data_and_models()
    except Exception as e:
        print(f"[Error] Genre Recommender preparation failed: {e}")
        return

    # 4.6. Initialize and fit Artist Similarity Recommender
    print(
        "\n[Step 2.6/3] Preparing Artist Similarity Recommender..."
    )
    artist_recommender = ArtistSimilarityRecommender(cleaned_data_path=train_path, model_dir=models_dir)
    try:
        artist_recommender.load_data_and_models()
    except Exception as e:
        print(f"[Error] Artist Recommender preparation failed: {e}")
        return

    # 4.7. Initialize and prepare Mood Recommendation Engine
    print(
        "\n[Step 2.7/3] Preparing Mood Recommendation Engine..."
    )
    mood_recommender = MoodRecommender(cleaned_data_path=train_path)
    try:
        mood_recommender.load_data_and_classify()
    except Exception as e:
        print(f"[Error] Mood Recommender preparation failed: {e}")
        return

    # 4.8. Initialize and prepare Weighted Hybrid Recommendation Engine
    print(
        "\n[Step 2.8/3] Preparing Weighted Hybrid Recommendation Engine..."
    )
    weighted_recommender = WeightedRecommender(cleaned_data_path=train_path, model_dir=models_dir)
    try:
        weighted_recommender.load_engines()
    except Exception as e:
        print(f"[Error] Weighted Hybrid Recommender preparation failed: {e}")
        return

    # Optionally load validation and test sets later for evaluation
    print(
        "\n[Evaluation Setup] Optionally loading validation and test sets for verification:"
    )
    try:
        if os.path.exists(val_path) and os.path.exists(test_path):
            val_loaded = pd.read_csv(val_path)
            test_loaded = pd.read_csv(test_path)
            print(
                f"  -> Validation set loaded successfully ({val_loaded.shape[0]} tracks, {val_loaded.shape[1]} columns)"
            )
            print(
                f"  -> Test set loaded successfully ({test_loaded.shape[0]} tracks, {test_loaded.shape[1]} columns)"
            )
    except Exception as e:
        print(f"  -> [Warning] Could not load validation/test sets: {e}")

    print("\n[Step 3/3] Ready for recommendations!")
    print("=" * 60)

    # Interactive query loop
    while True:
        # Show a few sample songs for inspiration
        # print("\n--- Sample Tracks Available in Database ---")
        sample_tracks = train_df.head(8)
        # for idx, row in sample_tracks.iterrows():
        #     print(f" • '{row['track_name']}' by {row['artists']}")

        print("\nRecommendation Engines:")
        print("  1. General Content-Based Recommender (across all genres)")
        print("  2. Genre-Specific Recommender (restricts recommendations to the same genre)")
        print("  3. Artist-Similarity Recommender (finds similar artists)")
        print("  4. Mood-Based Recommender (finds top songs for a mood)")
        print("  5. Weighted Hybrid Recommender (combines Audio, Artist, Genre, Popularity)")
        print("  6. Exit")

        engine_choice = input("\nEnter choice (1-6): ").strip()
        
        if engine_choice == "6" or engine_choice.lower() == "exit":
            print(
                "\nThank you for using Spotify Song Recommendation System! Keep rocking!"
            )
            print("=" * 60)
            break

        if engine_choice not in ["1", "2", "3", "4", "5"]:
            print("[Warning] Invalid choice. Please select 1, 2, 3, 4, 5, or 6.")
            continue

        if engine_choice == "3":
            query = input("\nEnter artist name for recommendations: ").strip()
        elif engine_choice == "4":
            query = input("\nEnter mood (Happy, Sad, Party, Chill) for recommendations: ").strip()
        else:
            query = input("\nEnter song name for recommendations: ").strip()

        if not query:
            if engine_choice == "3":
                print("[Warning] Artist name cannot be blank. Please try again.")
            elif engine_choice == "4":
                print("[Warning] Mood cannot be blank. Please try again.")
            else:
                print("[Warning] Song name cannot be blank. Please try again.")
            continue

        # Get recommendations using the selected engine
        if engine_choice == "1":
            rec_df, err = recommender.get_song_recommendations(query, top_n=5)
            note_str = "Note: Match index is calculated using mathematical Cosine Similarity of 9 audio features."
        elif engine_choice == "2":
            rec_df, err = genre_recommender.recommend_by_genre(query, top_n=5)
            note_str = "Note: Match index is calculated using Cosine Similarity of audio features within the same genre."
        elif engine_choice == "3":
            rec_df, err = artist_recommender.get_similar_artists(query, top_n=5)
            note_str = "Note: Match index is calculated using TF-IDF token overlap of artist names."
        elif engine_choice == "4":
            rec_df, err = mood_recommender.recommend_by_mood(query, top_n=5)
            note_str = "Note: Songs are the top most popular tracks classified under the requested mood."
        else:
            rec_df, err = weighted_recommender.get_weighted_recommendations(query, top_n=10)
            note_str = "Note: Match index combines 50% Audio Similarity, 20% Artist Similarity, 20% Genre Similarity, and 10% Popularity."

        if err:
            print(f"\n[Search Result] {err}")
            print("Tip: Check spelling or try one of the sample tracks shown above.")
            continue

        # Display recommendations in a high-fidelity tabular view
        print("\n" + "+" + "-" * 78 + "+")
        if engine_choice == "2" and not rec_df.empty:
            genre_name = rec_df.iloc[0]["track_genre"]
            title_str = f"RECOMMENDED SONGS IN GENRE '{genre_name.upper()}':"
        elif engine_choice == "3":
            title_str = f"RECOMMENDED ARTISTS SIMILAR TO '{query.upper()}':"
        elif engine_choice == "4":
            title_str = f"TOP POPULAR SONGS FOR MOOD '{query.upper()}':"
        elif engine_choice == "5":
            title_str = f"HYBRID WEIGHTED RECOMMENDATIONS FOR '{query.upper()}':"
        else:
            title_str = "RECOMMENDED SONGS SIMILAR TO:"
            
        print(f"| {title_str:<76} |") 
        print("+" + "-" * 78 + "+")

        if engine_choice == "3":
            print(
                f"| {'#':<3} | {'Artist Name':<53} | {'Match Match':<17} |"
            )
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                artist_str = row["artist"]
                if len(artist_str) > 51:
                    artist_str = artist_str[:48] + "..."
                match_pct = f"{row['similarity_score'] * 100:.1f}% Match"
                print(
                    f"| {idx + 1:<3} | {artist_str:<53} | {match_pct:<17} |"
                )
        elif engine_choice == "4":
            print(
                f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Popularity':<17} |"
            )
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                track_str = row["track_name"]
                if len(track_str) > 26:
                    track_str = track_str[:23] + "..."

                artist_str = row["artists"]
                if len(artist_str) > 20:
                    artist_str = artist_str[:17] + "..."

                pop_str = f"Popularity: {int(row['popularity'])}"

                print(
                    f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {pop_str:<17} |"
                )
        elif engine_choice == "5":
            print(
                f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Hybrid Match':<17} |"
            )
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                track_str = row["track_name"]
                if len(track_str) > 26:
                    track_str = track_str[:23] + "..."

                artist_str = row["artists"]
                if len(artist_str) > 20:
                    artist_str = artist_str[:17] + "..."

                match_pct = f"{row['final_score'] * 100:.1f}% Match"

                print(
                    f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {match_pct:<17} |"
                )
        else:
            print(
                f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Match Match':<17} |"
            )
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                track_str = row["track_name"]
                if len(track_str) > 26:
                    track_str = track_str[:23] + "..."

                artist_str = row["artists"]
                if len(artist_str) > 20:
                    artist_str = artist_str[:17] + "..."

                match_pct = f"{row['similarity_score'] * 100:.1f}% Match"

                print(
                    f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {match_pct:<17} |"
                )

        print("+" + "-" * 78 + "+")
        print(note_str)
        print("-" * 80)


if __name__ == "__main__":
    main()
