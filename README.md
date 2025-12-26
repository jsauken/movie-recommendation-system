# Movie Recommendation System Using Collaborative Filtering

## Overview
This project implements a movie recommendation system using collaborative filtering techniques.
The system analyzes historical user–movie rating data to predict user preferences and generate personalized movie recommendations.
A baseline model is compared with a matrix factorization-based collaborative filtering approach, and model performance is evaluated using RMSE.

## Dataset
MovieLens dataset:
- 100,000 ratings
- ~600 users
- ~9,000 movies
- ~3,600 tag applications
Source: GroupLens Research

## Methods
- Baseline model: Global mean rating
- Improved model: Matrix Factorization (SVD)
- Evaluation metric: RMSE

## How to Run (Windows)

1. Create a virtual environment:
python -m venv .venv

2. Allow script execution (PowerShell only):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

3. Activate the virtual environment:
.\.venv\Scripts\Activate.ps1

4. Install dependencies:
pip install -r requirements.txt

5. Run the project:
python src/train_recommender.py

## Output
- Model comparison results saved to results/model_comparison.csv
- RMSE comparison plot saved to results/model_comparison.png
- Sample movie recommendations saved to results/sample_recommendations.csv

