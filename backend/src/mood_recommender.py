import os
import pandas as pd


class MoodRecommender:
    """
    Mood-Based Recommendation Engine.

    NOTE: This system uses a deterministic, RULE-BASED classification logic
    based on audio feature thresholds (energy, tempo, valence). It is NOT a
    trained machine learning model.

    Rules:
    - Party: Fast-paced (scaled_tempo > 0.55) & High intensity (scaled_energy > 0.65)
    - Chill: Slow-paced (scaled_tempo < 0.45) & Calm intensity (scaled_energy < 0.40)
    - Happy: High positivity (scaled_valence > 0.50)
    - Sad: Lower positivity and low-to-moderate energy
    """
    def __init__(self, cleaned_data_path=None):
        if cleaned_data_path is None:
            if os.path.exists("data/cleaned_spotify.csv"):
                cleaned_data_path = "data/cleaned_spotify.csv"
            else:
                cleaned_data_path = "data/processed/train.csv"

        self.cleaned_data_path = cleaned_data_path
        self.df = None

    def load_data_and_classify(self):
        """
        Loads the preprocessed dataset and ensures every song has a mood classification.
        """
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.cleaned_data_path}. Please run preprocess.py first."
            )

        print(f"[MoodRecommender] Loading dataset from {self.cleaned_data_path}...")
        self.df = pd.read_csv(self.cleaned_data_path).reset_index(drop=True)

        if "mood" in self.df.columns:
            print("[MoodRecommender] 'mood' column already present in dataset.")
            return

        print("[MoodRecommender] Classifying songs into moods using rule-based thresholds...")
        self.df["mood"] = self.df.apply(self.classify_song, axis=1)

    def classify_song(self, row):
        """
        Classifies a single song into one of the four moods (Party, Chill, Happy, Sad).
        Uses scaled feature values [0, 1].
        """
        energy = row.get("scaled_energy", 0.5)
        tempo = row.get("scaled_tempo", 0.5)
        valence = row.get("scaled_valence", 0.5)

        # 1. Party: Fast tempo and high energy
        if energy > 0.65 and tempo > 0.55:
            return "Party"

        # 2. Chill: Quiet energy and slow tempo
        elif energy < 0.40 and tempo < 0.45:
            return "Chill"

        # 3. Happy: High positivity
        elif valence > 0.50:
            return "Happy"

        # 4. Sad: Lower positivity and lower energy
        else:
            return "Sad"

    def recommend_by_mood(self, mood, top_n=10):
        """
        Finds all songs matching the requested mood and returns the top_n songs
        sorted deterministically by popularity descending and track_id ascending.
        """
        if self.df is None:
            self.load_data_and_classify()

        search_mood = mood.strip().capitalize()
        valid_moods = ["Happy", "Sad", "Party", "Chill"]

        if search_mood not in valid_moods:
            return None, f"Invalid mood '{mood}'. Valid moods are: {', '.join(valid_moods)}"

        mood_filtered_df = self.df[self.df["mood"] == search_mood].copy()

        if mood_filtered_df.empty:
            return None, f"No songs found for mood '{search_mood}'."

        sort_cols = ["popularity"]
        ascending_flags = [False]
        if "track_id" in mood_filtered_df.columns:
            sort_cols.append("track_id")
            ascending_flags.append(True)

        top_songs = mood_filtered_df.sort_values(by=sort_cols, ascending=ascending_flags).head(top_n)
        return top_songs.reset_index(drop=True), None


# ==============================================================================
# Package-level Wrapper Function
# ==============================================================================

_global_mood_recommender = None


def recommend_by_mood(mood, top_n=10):
    """
    Module level helper function matching the recommend_by_mood requirement.
    Lazily initializes the global MoodRecommender instance.
    """
    global _global_mood_recommender
    if _global_mood_recommender is None:
        _global_mood_recommender = MoodRecommender()

    return _global_mood_recommender.recommend_by_mood(mood, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("           MOOD RECOMMENDER TESTING BLOCK            ")
    print("=" * 60)

    recommender = MoodRecommender()
    try:
        recommender.load_data_and_classify()
        print("\nSong Mood Classification Distribution:")
        print(recommender.df["mood"].value_counts())

        test_moods = ["Happy", "Sad", "Party", "Chill"]
        for mood in test_moods:
            print(f"\n--- Top 5 Songs for Mood: '{mood}' ---")
            rec_df, err = recommend_by_mood(mood, top_n=5)
            if err:
                print(f"Error: {err}")
            else:
                cols_to_print = [c for c in ["track_name", "artists", "popularity", "mood"] if c in rec_df.columns]
                print(rec_df[cols_to_print].to_string(index=False))
    except Exception as e:
        print(f"\n[Execution Error] {e}")
    print("=" * 60)
