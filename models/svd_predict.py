from utils.svd_utils import create_user_item_matrix, apply_svd, predict_ratings
import pandas as pd
import numpy as np


# Function to recommend top K movies for a given user
def recommend_top_k_movies_svd(user_id, k=10):
    # Load movie titles for recommendation
    movie_titles = pd.read_csv('../data/raw/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1])
    movie_titles.columns = ['movie_id', 'title']

    # Load and prepare the data
    ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_file = pd.read_csv('../data/raw/u.data', sep='\t', names=ratings_columns, encoding='latin-1')
    user_item_matrix_csr, user_item_matrix_df = create_user_item_matrix('../data/raw/u.data')

    # Apply SVD
    latent_factors = 10  # You can adjust the number of latent factors
    u, sigma, vt = apply_svd(user_item_matrix_csr, latent_factors)

    # Predict ratings
    all_movies = np.arange(1, max(user_item_matrix_df.columns) + 1)
    predicted_ratings_df = predict_ratings(u, sigma, vt, user_item_matrix_df, all_movies)

    # Check if user exists in predicted ratings
    if user_id not in predicted_ratings_df.index:
        return "User not found in the data."

    # Get the user's predictions
    user_ratings = predicted_ratings_df.loc[user_id]

    # Filter out movies the user has already rated
    user_interactions = ratings_file[ratings_file['user_id'] == user_id]['movie_id']
    user_ratings = user_ratings[~user_ratings.index.isin(user_interactions)]

    # Get top K movie IDs with the highest predicted rating
    top_k_movies = user_ratings.sort_values(ascending=False).head(k).index

    # Map movie IDs to titles
    top_k_movie_titles = movie_titles[movie_titles['movie_id'].isin(top_k_movies)]

    print("Top {} movie recommendations for user {}:".format(k, user_id))
    for id, title in zip(top_k_movie_titles['movie_id'], top_k_movie_titles['title']):
        print("Movie ID: {}, Title: {}".format(id, title))

    return top_k_movie_titles
