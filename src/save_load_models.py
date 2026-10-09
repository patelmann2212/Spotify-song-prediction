import os
import sys
import pickle

# Ensure root of project is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_all_models(models_dict, save_dir="models"):
    """
    Saves recommendation system models to disk using pickle.

    Parameters:
    - models_dict: Dictionary containing fitted model objects:
      {
        'scaler': MinMaxScaler object (fitted strictly on training set),
        'vectorizer': TfidfVectorizer object (for artists),
        'audio_similarity': numpy array (Cosine similarity of top audio tracks),
        'artist_similarity': dict or sparse matrix (Artist similarity state),
        'genre_similarity': dict or array (Genre profiles and similarity state)
      }
    - save_dir: Directory where pickle files will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("=== SAVING RECOMMENDATION SYSTEM MODELS ===")

    # 1. Save Scaler (MinMaxScaler)
    if "scaler" in models_dict and models_dict["scaler"] is not None:
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(models_dict["scaler"], f)
        print(f"[Save] Scaler saved to: {scaler_path}")

    # 2. Save TF-IDF Vectorizer
    if "vectorizer" in models_dict and models_dict["vectorizer"] is not None:
        vectorizer_path = os.path.join(save_dir, "artist_vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(models_dict["vectorizer"], f)
        print(f"[Save] TF-IDF Vectorizer saved to: {vectorizer_path}")

    # 3. Save Audio Similarity Matrix
    if "audio_similarity" in models_dict and models_dict["audio_similarity"] is not None:
        audio_sim_path = os.path.join(save_dir, "similarity.pkl")
        with open(audio_sim_path, "wb") as f:
            pickle.dump(models_dict["audio_similarity"], f)
        print(f"[Save] Audio Similarity Matrix saved to: {audio_sim_path}")

    # 4. Save Artist Similarity State
    if "artist_similarity" in models_dict and models_dict["artist_similarity"] is not None:
        artist_sim_path = os.path.join(save_dir, "artist_similarity.pkl")
        with open(artist_sim_path, "wb") as f:
            pickle.dump(models_dict["artist_similarity"], f)
        print(f"[Save] Artist Similarity Matrix/State saved to: {artist_sim_path}")

    # 5. Save Genre Similarity State
    if "genre_similarity" in models_dict and models_dict["genre_similarity"] is not None:
        genre_sim_path = os.path.join(save_dir, "genre_recommender.pkl")
        with open(genre_sim_path, "wb") as f:
            pickle.dump(models_dict["genre_similarity"], f)
        print(f"[Save] Genre Similarity State saved to: {genre_sim_path}")

    # 6. Save KMeans Model and Scaler
    if "kmeans_model" in models_dict and models_dict["kmeans_model"] is not None:
        import joblib
        kmeans_path = os.path.join(save_dir, "kmeans_model.pkl")
        joblib.dump(models_dict["kmeans_model"], kmeans_path)
        print(f"[Save] KMeans Model saved to: {kmeans_path}")

    if "kmeans_scaler" in models_dict and models_dict["kmeans_scaler"] is not None:
        import joblib
        kmeans_scaler_path = os.path.join(save_dir, "kmeans_scaler.pkl")
        joblib.dump(models_dict["kmeans_scaler"], kmeans_scaler_path)
        print(f"[Save] KMeans Scaler saved to: {kmeans_scaler_path}")

    print("All models serialized successfully!\n")


def load_all_models(save_dir="models"):
    """
    Loads all saved recommendation models from disk.

    Returns:
    - Dict of loaded models:
      {
        'scaler': Loaded MinMaxScaler (fitted on train split),
        'vectorizer': Loaded TfidfVectorizer,
        'audio_similarity': Loaded audio similarity matrix,
        'artist_similarity': Loaded artist similarity matrix/dict,
        'genre_similarity': Loaded genre similarity matrix/dict
      }
    """
    print("=== LOADING RECOMMENDATION SYSTEM MODELS ===")
    loaded_models = {}

    # 1. Load Scaler
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            loaded_models["scaler"] = pickle.load(f)
        print(f"[Load] Scaler loaded from: {scaler_path}")
    else:
        print(f"[Warning] Scaler pickle file not found at {scaler_path}")

    # 2. Load TF-IDF Vectorizer
    vectorizer_path = os.path.join(save_dir, "artist_vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            loaded_models["vectorizer"] = pickle.load(f)
        print(f"[Load] TF-IDF Vectorizer loaded from: {vectorizer_path}")

    # 3. Load Audio Similarity Matrix
    audio_sim_path = os.path.join(save_dir, "similarity.pkl")
    if os.path.exists(audio_sim_path):
        with open(audio_sim_path, "rb") as f:
            loaded_models["audio_similarity"] = pickle.load(f)
        print(f"[Load] Audio Similarity Matrix loaded from: {audio_sim_path}")

    # 4. Load Artist Similarity Matrix / bundled dictionary
    artist_sim_path = os.path.join(save_dir, "artist_similarity.pkl")
    if os.path.exists(artist_sim_path):
        with open(artist_sim_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                loaded_models["artist_similarity"] = data.get("tfidf_matrix")
                if "vectorizer" not in loaded_models or loaded_models["vectorizer"] is None:
                    loaded_models["vectorizer"] = data.get("vectorizer")
                print(f"[Load] Artist Similarity tfidf_matrix & Vectorizer extracted from bundled dictionary: {artist_sim_path}")
            else:
                loaded_models["artist_similarity"] = data
                print(f"[Load] Artist Similarity Matrix loaded from: {artist_sim_path}")

    # 5. Load Genre Similarity Matrix / bundled dictionary
    genre_sim_path = os.path.join(save_dir, "genre_recommender.pkl")
    if os.path.exists(genre_sim_path):
        with open(genre_sim_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                loaded_models["genre_similarity"] = data.get("genre_similarity_matrix")
                print(f"[Load] Genre Similarity Matrix extracted from bundled dictionary: {genre_sim_path}")
            else:
                loaded_models["genre_similarity"] = data
                print(f"[Load] Genre Similarity Matrix loaded from: {genre_sim_path}")

    # 6. Load KMeans Model and Scaler
    kmeans_path = os.path.join(save_dir, "kmeans_model.pkl")
    if os.path.exists(kmeans_path):
        import joblib
        loaded_models["kmeans_model"] = joblib.load(kmeans_path)
        print(f"[Load] KMeans Model loaded from: {kmeans_path}")

    kmeans_scaler_path = os.path.join(save_dir, "kmeans_scaler.pkl")
    if os.path.exists(kmeans_scaler_path):
        import joblib
        loaded_models["kmeans_scaler"] = joblib.load(kmeans_scaler_path)
        print(f"[Load] KMeans Scaler loaded from: {kmeans_scaler_path}")

    print("All available models deserialized successfully!\n")
    return loaded_models


if __name__ == "__main__":
    print("============================================================")
    print("       MODEL SERIALIZATION & DESERIALIZATION TESTING        ")
    print("============================================================")
    models = load_all_models()
    if models:
        print(f"Loaded {len(models)} model components successfully.")
    else:
        print("Note: Run main.py or preprocess.py first to generate model files on disk.")
    print("============================================================")
