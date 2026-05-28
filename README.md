# 🎵 Spotify Song Recommendation System (Phase-1)

A beginner-friendly **Content-Based Filtering Song Recommendation System** written in Python. This project calculates the mathematical similarity between songs based on their structural audio features (e.g., danceability, energy, tempo, acousticness) and recommends the most acoustically similar songs to the user.

---

## 🚀 Features
- **Pure Python & Standard ML Stack**: Built using `pandas`, `numpy`, and `scikit-learn`.
- **Cosine Similarity Engine**: Computes high-fidelity multi-dimensional vector similarities.
- **MinMax Scaling Pipeline**: Normalizes raw audio features (e.g., bringing tempo and loudness onto a uniform scale) for fair distance calculations.
- **Robust Out-of-the-Box Setup**: If your raw dataset is not present, the system automatically generates a curated mock dataset containing iconic Pop, Rock, and Acoustic songs so you can run it instantly.
- **Beautiful Console Interface**: Interactive Command-Line Interface showing percentage matches with high readability.

---

## 📂 Project Folder Structure

```text
Spotify-song-prediction/
│
├── data/
│   ├── raw/
│   │   └── spotify_tracks.csv          # Place your raw Spotify tracks dataset here
│   └── processed/
│       └── cleaned_tracks.csv          # Preprocessed, deduplicated and scaled data
│
├── src/
│   ├── __init__.py                     # Marks src as a package
│   ├── generate_dummy_data.py          # Auto-generates mock tracks if the raw CSV is missing
│   ├── preprocess.py                   # Data cleaning, null handling, and MinMaxScaler pipeline
│   ├── recommend.py                    # Computes cosine similarity matrices & queries recommendations
│   └── main.py                         # CLI controller and interactive song matching loop
│
├── .gitignore                          # Standard git exclusions (venv, python cache, OS files)
├── README.md                           # This guide!
└── requirements.txt                    # Project dependencies (pandas, numpy, scikit-learn)
```

---

## ⚙️ Setup and Installation

Follow these steps to set up the project on your local machine:

### 1. Set Up a Virtual Environment (Recommended)
Virtual environments keep your project's dependencies isolated from your global system.

**On Windows (PowerShell/CMD):**
```powershell
# Create virtual environment named '.venv'
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

To boot up the interactive recommendation engine, simply run the `main.py` entrypoint script:

```bash
python src/main.py
```

### What happens when you run it?
1. **Mock Data Check**: If `data/raw/spotify_tracks.csv` is missing, it will automatically populate a diverse dataset of 15 classic songs.
2. **Preprocessing**: The raw columns are cleaned, duplicate songs are removed, and numeric features are scaled using `MinMaxScaler`.
3. **Similarity Training**: The system builds a Cosine Similarity matrix.
4. **Interactive Prompt**: You can search for songs (like "Blinding Lights" or "Ocean Eyes") and get top recommendations!

---

## 🧠 Behind the Scenes: How Content-Based Filtering Works

This recommendation system uses **Content-Based Filtering**. It recommends songs to a user by matching the "content" (in this case, audio properties) of a target song.

### 1. Scaling the Audio Features
Different audio features have vastly different ranges. For example:
- `tempo` ranges from **60 to 200 BPM**.
- `acousticness` ranges from **0.0 to 1.0**.
- `loudness` ranges from **-60 to 0 dB**.

If we compute similarities directly, features with larger scales like `tempo` will dominate the math. We resolve this by applying a `MinMaxScaler` which bounds every feature exactly between `0.0` and `1.0`:
$$\text{Scaled Value} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### 2. Calculating Similarity with Cosine Similarity
We represent every song as a multi-dimensional vector of 9 audio features:
$$\vec{u} = [\text{danceability}, \text{energy}, \dots, \text{tempo}]$$

To measure the similarity between song $\vec{A}$ and song $\vec{B}$, we compute the **Cosine Similarity** (the cosine of the angle $\theta$ between the two vectors in 9D space):

$$\text{Similarity}(A, B) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **1.0 (100% Match)**: The songs are perfectly similar in their audio characteristics.
- **0.0 (0% Match)**: The songs are completely dissimilar.

---

## 🛠️ VS Code Extensions for Better Development
To make editing and exploring this project easier, install the following VS Code Extensions:
1. **Python** (by Microsoft): Syntax highlighting, linting, and environment selection.
2. **Pylance** (by Microsoft): Fast and feature-rich Python language support.
3. **CSV to Table** (by Zdravko B.): Easily inspect the raw or preprocessed CSV data files as clean, interactive spreadsheets.
4. **Markdown All in One** (by Yu Zhang): Enhanced editing tools and live preview for this README.
