import os
import sys
import pickle

# Ensure the root of the project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_all_models(models_dict, save_dir="models"):
    """
    Demonstrates how to save all five Phase-2 recommendation models using pickle.
    
    Parameters:
    - models_dict: Dictionary containing the fitted model objects:
      {
        'scaler': MinMaxScaler object,
        'vectorizer': TfidfVectorizer object (for artists),
        'audio_similarity': numpy array (Cosine similarity of audio features),
        'artist_similarity': sparse TF-IDF matrix / array (Cosine similarity of artists),
        'genre_similarity': numpy array (Cosine similarity of genre profiles)
      }
    - save_dir: Directory where the pickle files will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("=== SAVING PHASE-2 MODELS ===")
    
    # 1. Save Scaler (MinMaxScaler)
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(models_dict["scaler"], f)
    print(f"[Save] Scaler saved to: {scaler_path}")
    
    # 2. Save TF-IDF Vectorizer
    vectorizer_path = os.path.join(save_dir, "artist_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(models_dict["vectorizer"], f)
    print(f"[Save] TF-IDF Vectorizer saved to: {vectorizer_path}")
    
    # 3. Save Audio Similarity Matrix
    audio_sim_path = os.path.join(save_dir, "similarity.pkl")
    with open(audio_sim_path, "wb") as f:
        pickle.dump(models_dict["audio_similarity"], f)
    print(f"[Save] Audio Similarity Matrix saved to: {audio_sim_path}")
    
    # 4. Save Artist Similarity Matrix
    artist_sim_path = os.path.join(save_dir, "artist_similarity.pkl")
    with open(artist_sim_path, "wb") as f:
        pickle.dump(models_dict["artist_similarity"], f)
    print(f"[Save] Artist Similarity Matrix saved to: {artist_sim_path}")
    
    # 5. Save Genre Similarity Matrix
    genre_sim_path = os.path.join(save_dir, "genre_recommender.pkl")
    with open(genre_sim_path, "wb") as f:
        pickle.dump(models_dict["genre_similarity"], f)
    print(f"[Save] Genre Similarity Matrix saved to: {genre_sim_path}")
    
    print("All models serialized successfully!\n")


def load_all_models(save_dir="models"):
    """
    Demonstrates how to load all five Phase-2 recommendation models using pickle.
    
    Returns:
    - Dict of loaded models:
      {
        'scaler': Loaded MinMaxScaler,
        'vectorizer': Loaded TfidfVectorizer,
        'audio_similarity': Loaded audio similarity matrix,
        'artist_similarity': Loaded artist similarity matrix,
        'genre_similarity': Loaded genre similarity matrix
      }
    """
    print("=== LOADING PHASE-2 MODELS ===")
    loaded_models = {}
    
    # 1. Load Scaler
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            loaded_models["scaler"] = pickle.load(f)
        print(f"[Load] Scaler loaded from: {scaler_path}")
    else:
        print(f"[Warning] Scaler pickle file not found at {scaler_path}")
        
    # 2. Load TF-IDF Vectorizer (for artist names)
    vectorizer_path = os.path.join(save_dir, "artist_vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            loaded_models["vectorizer"] = pickle.load(f)
        print(f"[Load] TF-IDF Vectorizer loaded from: {vectorizer_path}")
    else:
        # Note: If separate vectorizer.pkl is not created yet, 
        # it is often bundled inside artist_similarity.pkl or can be fit on the fly.
        print(f"[Load] TF-IDF Vectorizer standalone file not found, trying bundled dictionary...")
    
    # 3. Load Audio Similarity Matrix
    audio_sim_path = os.path.join(save_dir, "similarity.pkl")
    if os.path.exists(audio_sim_path):
        with open(audio_sim_path, "rb") as f:
            loaded_models["audio_similarity"] = pickle.load(f)
        print(f"[Load] Audio Similarity Matrix loaded from: {audio_sim_path}")
    else:
        print(f"[Warning] Audio Similarity Matrix pickle file not found at {audio_sim_path}")
        
    # 4. Load Artist Similarity Matrix (or bundled dictionary)
    artist_sim_path = os.path.join(save_dir, "artist_similarity.pkl")
    if os.path.exists(artist_sim_path):
        with open(artist_sim_path, "rb") as f:
            data = pickle.load(f)
            # Handle if it was saved as a bundled dictionary (as in artist_recommender.py)
            if isinstance(data, dict):
                loaded_models["artist_similarity"] = data.get("tfidf_matrix")
                loaded_models["vectorizer"] = data.get("vectorizer")
                print(f"[Load] Artist Similarity tfidf_matrix & Vectorizer extracted from bundled dictionary: {artist_sim_path}")
            else:
                loaded_models["artist_similarity"] = data
                print(f"[Load] Artist Similarity Matrix loaded from: {artist_sim_path}")
    else:
        print(f"[Warning] Artist Similarity Matrix pickle file not found at {artist_sim_path}")
        
    # 5. Load Genre Similarity Matrix (or bundled dictionary)
    genre_sim_path = os.path.join(save_dir, "genre_recommender.pkl")
    if os.path.exists(genre_sim_path):
        with open(genre_sim_path, "rb") as f:
            data = pickle.load(f)
            # Handle if it was saved as a bundled dictionary (as in genre_recommender.py)
            if isinstance(data, dict):
                loaded_models["genre_similarity"] = data.get("genre_similarity_matrix")
                print(f"[Load] Genre Similarity Matrix extracted from bundled dictionary: {genre_sim_path}")
            else:
                loaded_models["genre_similarity"] = data
                print(f"[Load] Genre Similarity Matrix loaded from: {genre_sim_path}")
    else:
        print(f"[Warning] Genre Similarity Matrix pickle file not found at {genre_sim_path}")
        
    print("All available models deserialized successfully!\n")
    return loaded_models


if __name__ == "__main__":
    # Test script block to verify serialization and deserialization
    print("============================================================")
    print("       MODEL SERIALIZATION & DESERIALIZATION TESTING        ")
    print("============================================================")
    
    # 1. Attempt to load current models from disk (demonstrating loading)
    models = load_all_models()
    
    # 2. Re-save them (demonstrating saving)
    if models:
        # Check if vectorizer is present
        if "vectorizer" not in models or models["vectorizer"] is None:
            # Create a mock vectorizer if none exists for demonstration
            from sklearn.feature_extraction.text import TfidfVectorizer
            mock_vec = TfidfVectorizer()
            mock_vec.fit(["lata mangeshkar", "kishore kumar", "the beatles"])
            models["vectorizer"] = mock_vec
            
        # Ensure we have all items
        save_dict = {
            "scaler": models.get("scaler"),
            "vectorizer": models.get("vectorizer"),
            "audio_similarity": models.get("audio_similarity"),
            "artist_similarity": models.get("artist_similarity"),
            "genre_similarity": models.get("genre_similarity")
        }
        
        # Save them back (will overwrite or create artist_vectorizer.pkl)
        save_all_models(save_dict)
        
        # Re-load to make sure everything works
        reloaded = load_all_models()
        
    else:
        print("Note: Run main.py first to generate the model files on disk.")
    print("============================================================")
