import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
df = pd.read_csv("dataset.csv")


# Keep important columns
df = df[[
    "track_name",
    "artists",
    "track_genre"
]]


# Remove null values
df.dropna(inplace=True)


# Remove duplicates
df.drop_duplicates(inplace=True)


# Reduce dataset size for RAM safety
df = df.sample(5000, random_state=42)

df.reset_index(drop=True, inplace=True)


# Create combined features
df["combined"] = (
    df["artists"].astype(str)
    + " "
    + df["track_genre"].astype(str)
)


# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words="english")

matrix = tfidf.fit_transform(df["combined"])


# Similarity matrix
similarity = cosine_similarity(matrix)


# Save files
joblib.dump(similarity, "model/similarity.pkl")

joblib.dump(df, "model/songs.pkl")


print("Model files created successfully.")