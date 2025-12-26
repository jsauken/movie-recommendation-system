# Movie Recommendation System Using Collaborative Filtering

## Overview
This project implements a movie recommendation system using collaborative filtering techniques. The system analyzes historical user–movie rating data to predict user preferences and generate personalized movie recommendations. A baseline model is compared with a matrix factorization-based collaborative filtering approach, and model performance is evaluated using RMSE.

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

## How to Run
1. Install dependencies:
```bash
python -m venv .venv
pip install -r requirements.txt


