import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def create_user_item_matrix(data_path):
    """
    Create a user-item matrix from the raw data

    Parameters:
    data_path (str): Path to the raw data

    Returns:
    user_item_matrix_csr (csr_matrix): User-item matrix in CSR format
    user_item_matrix_df (DataFrame): User-item matrix in DataFrame format
    """
    data = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return csr_matrix(user_item_matrix.values), user_item_matrix


def apply_svd(user_item_matrix_csr, k):
    """
    Apply Singular Value Decomposition (SVD) to the user-item matrix

    Parameters:
    user_item_matrix_csr (csr_matrix): User-item matrix in CSR format
    k (int): Number of latent factors

    Returns:
    u (ndarray): Left singular vectors
    sigma (ndarray): Singular values
    vt (ndarray): Right singular vectors
    """
    u, s, vt = svds(user_item_matrix_csr, k=k)
    sigma = np.diag(s)
    return u, sigma, vt


def predict_ratings(u, sigma, vt, user_item_matrix_df, all_movies):
    """
    Predict ratings for all users and items

    Parameters:
    u (ndarray): Left singular vectors
    sigma (ndarray): Singular values
    vt (ndarray): Right singular vectors
    user_item_matrix_df (DataFrame): User-item matrix in DataFrame format
    all_movies (ndarray): List of all movies

    Returns:
    predicted_ratings_df (DataFrame): Predicted ratings for all users and items
    """
    predicted_ratings = np.dot(np.dot(u, sigma), vt) + user_item_matrix_df.mean(axis=1).values.reshape(-1, 1)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix_df.index, columns=user_item_matrix_df.columns)

    # Reindex the DataFrame to include all movies
    predicted_ratings_df = predicted_ratings_df.reindex(columns=all_movies, fill_value=np.mean(predicted_ratings_df.values))
    return predicted_ratings_df
