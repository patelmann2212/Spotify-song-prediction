import os
import pandas as pd

class MoodRecommender:
    """
    Mood-Based Recommendation System.
    
    This system classifies tracks into one of four moods:
    - Happy: High positivity (valence) and moderate-to-high energy.
    - Sad: Low positivity (valence) and low energy.
    - Party: High energy and high tempo (speed).
    - Chill: Low energy, low tempo, and moderate-to-high positivity.
    
    It classifies all tracks and provides recommendations for a selected mood
    by returning the top songs sorted by popularity.
    """
    def __init__(self, cleaned_data_path="data/processed/train.csv"):
        self.cleaned_data_path = cleaned_data_path
        self.df = None

    def load_data_and_classify(self):
        """
        Loads the preprocessed dataset and classifies every song into a mood.
        Saves the classified dataset to disk so we don't have to reclassify every time.
        """
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(
                f"Training dataset not found at {self.cleaned_data_path}. Please run main.py first."
            )
        
        print(f"[MoodRecommender] Loading dataset from {self.cleaned_data_path}...")
        self.df = pd.read_csv(self.cleaned_data_path)
        
        # Check if we already have the 'mood' column in the dataset
        if "mood" in self.df.columns:
            print("[MoodRecommender] 'mood' column already exists in dataset.")
            return

        print("[MoodRecommender] Classifying songs into moods using valence, energy, and tempo...")
        
        # Apply the classification function to every song (row) in the DataFrame
        self.df["mood"] = self.df.apply(self.classify_song, axis=1)
        
        # Save the dataset back to include the new 'mood' column
        self.df.to_csv(self.cleaned_data_path, index=False)
        print(f"[MoodRecommender] Classified dataset saved to {self.cleaned_data_path}")

    def classify_song(self, row):
        """
        Classifies a single song into one of the four moods (Happy, Sad, Party, Chill).
        
        We use the scaled features because they are normalized between 0 and 1,
        making the threshold rules uniform and easy to understand:
        - scaled_valence: measures musical positiveness (0 = sad/angry, 1 = happy/euphoric)
        - scaled_energy: measures intensity and activity (0 = calm/quiet, 1 = loud/energetic)
        - scaled_tempo: measures beats per minute (0 = slow, 1 = fast)
        """
        valence = row.get("scaled_valence", 0.5)
        energy = row.get("scaled_energy", 0.5)
        tempo = row.get("scaled_tempo", 0.5)
        
        # 1. Party: Fast-paced (high tempo) and highly energetic (high energy)
        if energy > 0.65 and tempo > 0.55:
            return "Party"
        
        # 2. Chill: Quiet (low energy) and slow-paced (low tempo)
        elif energy < 0.40 and tempo < 0.45:
            return "Chill"
        
        # 3. Happy: Highly positive vibe (high valence) and moderate/high energy
        elif valence > 0.50:
            return "Happy"
        
        # 4. Sad: Default category for low positivity and low energy music
        else:
            return "Sad"

    def recommend_by_mood(self, mood, top_n=10):
        """
        Finds all songs matching the requested mood and returns the top_n songs
        sorted by popularity.
        """
        # Ensure data is loaded and classified
        if self.df is None:
            self.load_data_and_classify()
            
        # Clean the input mood and make it case-insensitive
        search_mood = mood.strip().capitalize()
        
        valid_moods = ["Happy", "Sad", "Party", "Chill"]
        if search_mood not in valid_moods:
            return None, f"Invalid mood '{mood}'. Valid moods are: {', '.join(valid_moods)}"
            
        # Filter the DataFrame to keep only songs that belong to the requested mood
        mood_filtered_df = self.df[self.df["mood"] == search_mood]
        
        if mood_filtered_df.empty:
            return None, f"No songs found for mood '{search_mood}'."
            
        # Sort the filtered songs by popularity in descending order (highest first)
        top_songs = mood_filtered_df.sort_values(by="popularity", ascending=False).head(top_n)
        
        # Reset the index for a clean output presentation
        top_songs = top_songs.reset_index(drop=True)
        
        return top_songs, None


# ==============================================================================
# Package-level Wrapper Function (Required by prompt)
# ==============================================================================

_global_mood_recommender = None

def recommend_by_mood(mood, top_n=10):
    """
    Module level helper function matching the recommend_by_mood(mood) requirement.
    Lazily initializes the global MoodRecommender instance.
    """
    global _global_mood_recommender
    if _global_mood_recommender is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "processed", "train.csv")
        _global_mood_recommender = MoodRecommender(cleaned_data_path=train_path)
        
    return _global_mood_recommender.recommend_by_mood(mood, top_n)


if __name__ == "__main__":
    print("=" * 60)
    print("           MOOD RECOMMENDER TESTING BLOCK            ")
    print("=" * 60)
    
    recommender = MoodRecommender()
    try:
        # Load and run classifications
        recommender.load_data_and_classify()
        
        # Print classification distribution
        print("\nSong Mood Classification Distribution:")
        print(recommender.df["mood"].value_counts())
        
        # Test recommendations for each mood
        test_moods = ["Happy", "Sad", "Party", "Chill"]
        for mood in test_moods:
            print(f"\n--- Top 5 Songs for Mood: '{mood}' ---")
            rec_df, err = recommend_by_mood(mood, top_n=5)
            if err:
                print(f"Error: {err}")
            else:
                # Select only relevant columns to print nicely
                cols_to_print = ["track_name", "artists", "popularity", "mood"]
                print(rec_df[cols_to_print].to_string(index=False))
                
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        print("Note: Make sure data/processed/train.csv exists by running main.py first.")
    print("=" * 60)
