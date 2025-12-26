import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split

from config import (
    RATINGS_PATH, MOVIES_PATH, TAGS_PATH,
    DATA_PROCESSED_DIR, RESULTS_DIR,
    CLEANED_RATINGS_PATH,
    MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
    RANDOM_STATE
)
from data_utils import load_raw_data, preprocess_ratings, merge_ratings_movies
    

def run_eda(ratings_movies: pd.DataFrame):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Rating distribution
    plt.figure()
    plt.hist(ratings_movies["rating"], bins=10)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rating_distribution.png")
    plt.close()

    # 2) Ratings per user
    plt.figure()
    ratings_per_user = ratings_movies.groupby("userId").size()
    plt.hist(ratings_per_user, bins=30)
    plt.title("Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Users")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ratings_per_user.png")
    plt.close()

    # 3) Top 10 most-rated movies
    plt.figure()
    top_movies = ratings_movies["title"].value_counts().head(10)
    top_movies.plot(kind="barh")
    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_10_most_rated_movies.png")
    plt.close()


def baseline_global_mean_rmse(ratings_movies: pd.DataFrame) -> float:
    train_df, test_df = train_test_split(
        ratings_movies, test_size=0.2, random_state=RANDOM_STATE
    )

    global_mean = train_df["rating"].mean()
    preds = np.full(shape=len(test_df), fill_value=global_mean, dtype=float)

    rmse = np.sqrt(mean_squared_error(test_df["rating"], preds))
    return rmse


def train_svd_rmse(ratings_clean: pd.DataFrame) -> float:
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_clean[["userId", "movieId", "rating"]], reader)

    trainset, testset = surprise_train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE
    )

    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )
    model.fit(trainset)

    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)

    return rmse, model


def recommend_movies(model, ratings_clean, movies, user_id: int, n: int = 10) -> pd.DataFrame:
    user_rated = set(ratings_clean.loc[ratings_clean["userId"] == user_id, "movieId"])
    all_movies = set(ratings_clean["movieId"].unique())
    candidates = list(all_movies - user_rated)

    predictions = [model.predict(user_id, mid) for mid in candidates]
    top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    rec_movie_ids = [p.iid for p in top_preds]
    rec_scores = [p.est for p in top_preds]

    recs = movies[movies["movieId"].isin(rec_movie_ids)].copy()
    recs["pred_rating"] = recs["movieId"].map(dict(zip(rec_movie_ids, rec_scores)))
    recs = recs.sort_values("pred_rating", ascending=False)

    return recs[["movieId", "title", "genres", "pred_rating"]]


def main():
    # 1. Load data
    # 2. Preprocess
    # 3. EDA
    # 4. Baseline model
    # 5. Improved model (SVD)
    # 6. Compare models
    # 7. Recommendations

    # Create dirs
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    ratings, movies, tags = load_raw_data(RATINGS_PATH, MOVIES_PATH, TAGS_PATH)

    # Preprocess
    ratings_clean = preprocess_ratings(
        ratings,
        min_user_ratings=MIN_USER_RATINGS,
        min_movie_ratings=MIN_MOVIE_RATINGS
    )

    # Save cleaned dataset
    ratings_clean.to_csv(CLEANED_RATINGS_PATH, index=False)

    # Merge for EDA
    ratings_movies = merge_ratings_movies(ratings_clean, movies)

    # EDA (saves plots to results/)
    run_eda(ratings_movies)

    # Baseline
    rmse_baseline = baseline_global_mean_rmse(ratings_movies)

    # Improved model (SVD)
    rmse_svd, svd_model = train_svd_rmse(ratings_clean)

    # Save comparison
    results = pd.DataFrame({
        "Model": ["Baseline (Global Mean)", "SVD Collaborative Filtering"],
        "RMSE": [rmse_baseline, rmse_svd]
    })
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\n=== Model Results ===")
    print(results.to_string(index=False))

    plt.figure()
    plt.bar(results["Model"], results["RMSE"])
    plt.ylabel("RMSE (lower is better)")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "model_comparison.png")
    plt.show()

    # Example recommendations (save to csv)
    # Picking a random user
    example_user = random.choice(ratings_clean["userId"].unique())
    recs = recommend_movies(svd_model, ratings_clean, movies, user_id=example_user, n=10)
    recs.to_csv(RESULTS_DIR / "sample_recommendations.csv", index=False)

    print(f"\nTop recommendations for user {example_user}:")
    print(recs[["title", "pred_rating"]].head(10).to_string(index=False))
    print(f"\nTop recommendations for user {example_user}:")
    print(recs[["title", "pred_rating"]].head(10).to_string(index=False))

    # -----------------------------
    # Interactive demo (ASK USER)
    # -----------------------------
    while True:
        min_user = int(ratings_clean["userId"].min())
        max_user = int(ratings_clean["userId"].max())
        print(f"\nYou can enter a userId between {min_user} and {max_user}")
        user_input = input("\nEnter userId for recommendations (or 'q' to quit): ").strip()

        if user_input.lower() == "q":
            print("Exiting recommendation demo.")
            break

        try:
            user_id = int(user_input)
        except ValueError:
            print("Please enter a valid numeric userId.")
            continue

        if user_id not in set(ratings_clean["userId"].unique()):
            print("User not found in dataset.")
            continue

        n_str = input("How many recommendations? (default 10): ").strip()
        n = int(n_str) if n_str.isdigit() else 10

        recs_user = recommend_movies(
            svd_model,
            ratings_clean,
            movies,
            user_id=user_id,
            n=n
        )

        print(recs_user[["title", "genres", "pred_rating"]].to_string(index=False))

if __name__ == "__main__":
    main()
