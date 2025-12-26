import pandas as pd

def load_raw_data(ratings_path, movies_path, tags_path=None):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path) if tags_path is not None else None
    return ratings, movies, tags

def preprocess_ratings(ratings, min_user_ratings=20, min_movie_ratings=20):
    """
    - drops duplicate (userId, movieId)
    - filters users/movies with few ratings to reduce sparsity/noise
    """
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"]).copy()

    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    ratings = ratings[
        ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index) &
        ratings["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)
    ].copy()

    return ratings

def merge_ratings_movies(ratings, movies):
    return ratings.merge(movies, on="movieId", how="left")
