from pathlib import Path

# Project root = folder that contains /src, /data, /results
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

RATINGS_PATH = DATA_RAW_DIR / "ratings.csv"
MOVIES_PATH = DATA_RAW_DIR / "movies.csv"
TAGS_PATH = DATA_RAW_DIR / "tags.csv"

CLEANED_RATINGS_PATH = DATA_PROCESSED_DIR / "ratings_cleaned.csv"

# Preprocessing thresholds (you can change these)
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 20

RANDOM_STATE = 42
