import os
import sys

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force terminal to use UTF-8 encoding on Windows to prevent UnicodeEncodeErrors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import pandas as pd
from src.preprocess import (
    load_data,
    preprocess_pipeline,
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
    print("      SPOTIFY SONG RECOMMENDATION SYSTEM (PHASE-1 & 2)  ")
    print("      Content-Based, Genre, Artist, Mood & Hybrid       ")
    print("=" * 60)

    # Define file paths
    raw_path = os.path.join("data", "raw", "spotify_tracks.csv")
    cleaned_catalog_path = os.path.join("data", "cleaned_spotify.csv")
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "validation.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    models_dir = os.path.join("models")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    similarity_matrix_path = os.path.join(models_dir, "similarity.pkl")

    # 1. Load or generate processed data
    needs_preprocessing = not (
        os.path.exists(cleaned_catalog_path)
        and os.path.exists(train_path)
        and os.path.exists(scaler_path)
    )

    if needs_preprocessing:
        if not os.path.exists(raw_path):
            print(f"[Error] Raw dataset 'spotify_tracks.csv' not found at: {raw_path}")
            print("Please place your Spotify tracks dataset CSV there and re-run.")
            return

        print("\n[Step 1/3] Preprocessing raw dataset with leak-free, group-aware split...")
        try:
            raw_df = load_data(raw_path)
            catalog_df, train_df, val_df, test_df, scaler = preprocess_pipeline(raw_df)

            # Save recommendation catalog and evaluation splits
            save_processed_data(catalog_df, cleaned_catalog_path)
            save_processed_data(train_df, train_path)
            save_processed_data(val_df, val_path)
            save_processed_data(test_df, test_path)
            save_scaler(scaler, scaler_path)
            print(f"  -> Preprocessing complete. Recommendation catalog saved to {cleaned_catalog_path}")
        except Exception as e:
            print(f"[Error] Preprocessing failed: {e}")
            return
    else:
        print(f"[Info] Loading searchable recommendation catalog from {cleaned_catalog_path}...")
        try:
            catalog_df = pd.read_csv(cleaned_catalog_path)
            print(f"  -> Loaded catalog: {catalog_df.shape[0]} tracks")
        except Exception as e:
            print(f"[Error] Failed to load catalog: {e}")
            return

    # 2. Initialize engines using the full recommendation catalog
    print("\n[Step 2/3] Preparing Recommendation Engines on searchable catalog...")

    # Engine 1: General Content-Based Recommender
    print("  -> Initializing Content-Based Recommender...")
    recommender = ContentBasedRecommender(catalog_df)
    try:
        recommender.fit(build_matrix=True)
        recommender.save_similarity_matrix(similarity_matrix_path)
    except Exception as e:
        print(f"[Error] Content-Based Recommender initialization failed: {e}")
        return

    # Engine 2: Genre-Specific Recommender
    print("  -> Initializing Genre Recommendation Engine...")
    genre_recommender = GenreRecommender(
        cleaned_data_path=cleaned_catalog_path, model_dir=models_dir
    )
    try:
        genre_recommender.load_data_and_models()
    except Exception as e:
        print(f"[Error] Genre Recommender initialization failed: {e}")
        return

    # Engine 3: Artist Similarity Recommender
    print("  -> Initializing Artist Similarity Recommender...")
    artist_recommender = ArtistSimilarityRecommender(
        cleaned_data_path=cleaned_catalog_path, model_dir=models_dir
    )
    try:
        artist_recommender.load_data_and_models()
    except Exception as e:
        print(f"[Error] Artist Recommender initialization failed: {e}")
        return

    # Engine 4: Mood Recommender
    print("  -> Initializing Mood Recommendation Engine...")
    mood_recommender = MoodRecommender(cleaned_data_path=cleaned_catalog_path)
    try:
        mood_recommender.load_data_and_classify()
    except Exception as e:
        print(f"[Error] Mood Recommender initialization failed: {e}")
        return

    # Engine 5: Weighted Hybrid Recommender
    print("  -> Initializing Weighted Hybrid Recommender...")
    weighted_recommender = WeightedRecommender(
        cleaned_data_path=cleaned_catalog_path, model_dir=models_dir
    )
    try:
        weighted_recommender.load_engines()
    except Exception as e:
        print(f"[Error] Weighted Hybrid Recommender initialization failed: {e}")
        return

    # Verification of evaluation datasets
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_rows = pd.read_csv(train_path, usecols=["track_id"]).shape[0]
        val_rows = pd.read_csv(val_path, usecols=["track_id"]).shape[0]
        test_rows = pd.read_csv(test_path, usecols=["track_id"]).shape[0]
        print(
            f"\n[Evaluation Sets Verified] Train: {train_rows} tracks | Val: {val_rows} tracks | Test: {test_rows} tracks (Zero group leakage)"
        )

    print("\n[Step 3/3] System Ready!")
    print("=" * 60)

    # Interactive query loop
    while True:
        print("\nRecommendation Engines:")
        print("  1. General Content-Based Recommender (across all genres)")
        print("  2. Genre-Specific Recommender (restricts recommendations to the same genre)")
        print("  3. Artist-Similarity Recommender (finds similar artists via token overlap)")
        print("  4. Mood-Based Recommender (finds top songs for a mood)")
        print("  5. Weighted Hybrid Recommender (combines Audio, Artist, Genre, Popularity)")
        print("  6. Exit")

        engine_choice = input("\nEnter choice (1-6): ").strip()

        if engine_choice == "6" or engine_choice.lower() == "exit":
            print("\nThank you for using Spotify Song Recommendation System! Keep rocking!")
            print("=" * 60)
            break

        if engine_choice not in ["1", "2", "3", "4", "5"]:
            print("[Warning] Invalid choice. Please select 1, 2, 3, 4, 5, or 6.")
            continue

        if engine_choice == "3":
            artist_query = input("\nEnter artist name: ").strip()
            if not artist_query:
                print("[Warning] Artist name cannot be blank. Please try again.")
                continue
            rec_df, err = artist_recommender.get_similar_artists(artist_query, top_n=5)
            note_str = "Note: Similarity is computed using TF-IDF token overlap of artist name strings."
        elif engine_choice == "4":
            mood_query = input("\nEnter mood (Happy, Sad, Party, Chill): ").strip()
            if not mood_query:
                print("[Warning] Mood cannot be blank. Please try again.")
                continue
            rec_df, err = mood_recommender.recommend_by_mood(mood_query, top_n=5)
            note_str = "Note: Songs are top popular tracks classified using deterministic rule-based audio thresholds."
        else:
            song_query = input("\nEnter song name: ").strip()
            if not song_query:
                print("[Warning] Song name cannot be blank. Please try again.")
                continue
            artist_input = input("Enter artist name (optional, press Enter to skip): ").strip()
            artist_filter = artist_input if artist_input else None

            if engine_choice == "1":
                rec_df, err = recommender.get_song_recommendations(song_query, artist_filter, top_n=5)
                note_str = "Note: Similarity is calculated using Cosine Similarity of 10 scaled audio features."
            elif engine_choice == "2":
                rec_df, err = genre_recommender.recommend_by_genre(song_query, artist_filter, top_n=5)
                note_str = "Note: Similarity is calculated using Cosine Similarity of audio features within the same genre."
            else:
                rec_df, err = weighted_recommender.get_weighted_recommendations(song_query, artist_filter, top_n=10)
                note_str = "Note: Combined score: 50% Audio Similarity + 20% Artist Similarity + 20% Genre Profile + 10% Popularity."

        if err:
            print(f"\n[Search Result] {err}")
            continue

        # Display recommendations in formatted view
        print("\n" + "+" + "-" * 78 + "+")
        if engine_choice == "2" and not rec_df.empty and "track_genre" in rec_df.columns:
            genre_name = rec_df.iloc[0]["track_genre"]
            title_str = f"RECOMMENDED SONGS IN GENRE '{genre_name.upper()}':"
        elif engine_choice == "3":
            title_str = f"RECOMMENDED ARTISTS SIMILAR TO '{artist_query.upper()}':"
        elif engine_choice == "4":
            title_str = f"TOP POPULAR SONGS FOR MOOD '{mood_query.upper()}':"
        elif engine_choice == "5":
            title_str = f"HYBRID WEIGHTED RECOMMENDATIONS FOR '{song_query.upper()}':"
        else:
            title_str = f"RECOMMENDED SONGS SIMILAR TO '{song_query.upper()}':"

        print(f"| {title_str:<76} |")
        print("+" + "-" * 78 + "+")

        if engine_choice == "3":
            print(f"| {'#':<3} | {'Artist Name':<53} | {'Match':<17} |")
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                artist_str = str(row["artist"])
                if len(artist_str) > 51:
                    artist_str = artist_str[:48] + "..."
                match_pct = f"{row['similarity_score'] * 100:.1f}% Match"
                print(f"| {idx + 1:<3} | {artist_str:<53} | {match_pct:<17} |")
        elif engine_choice == "4":
            print(f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Popularity':<17} |")
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                track_str = str(row["track_name"])[:26]
                artist_str = str(row["artists"])[:20]
                pop_str = f"Popularity: {int(row['popularity'])}"
                print(f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {pop_str:<17} |")
        elif engine_choice == "5":
            print(f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Hybrid Match':<17} |")
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                track_str = str(row["track_name"])[:26]
                artist_str = str(row["artists"])[:20]
                match_pct = f"{row['final_score'] * 100:.1f}% Match"
                print(f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {match_pct:<17} |")
        else:
            print(f"| {'#':<3} | {'Track Name':<28} | {'Artist':<22} | {'Audio Match':<17} |")
            print("+" + "-" * 78 + "+")
            for idx, row in rec_df.iterrows():
                track_str = str(row["track_name"])[:26]
                artist_str = str(row["artists"])[:20]
                match_pct = f"{row['similarity_score'] * 100:.1f}% Match"
                print(f"| {idx + 1:<3} | {track_str:<28} | {artist_str:<22} | {match_pct:<17} |")

        print("+" + "-" * 78 + "+")
        print(note_str)
        print("-" * 80)


if __name__ == "__main__":
    main()
