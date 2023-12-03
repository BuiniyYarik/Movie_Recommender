import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import MinMaxScaler


item_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']


def aggregate_genres(row):
    """
    Aggregate genres into a single column

    Parameters:
    row (Series): A row of the DataFrame

    Returns:
    str: A string of genres separated by a pipe (|)
    """
    return '|'.join([genre for genre in item_columns[6:] if row[genre] == 1])


def recommend_top_k_movies_lr(user_id, k=10):
    # Load movie titles for recommendation
    movie_titles = pd.read_csv('../data/raw/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1])
    movie_titles.columns = ['movie_id', 'title']

    # Load and prepare the data
    ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_file = pd.read_csv('../data/raw/u.data', sep='\t', names=ratings_columns, encoding='latin-1')

    # Load user and movie data
    users = pd.read_csv('../data/raw/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], encoding='latin-1')
    movies = pd.read_csv('../data/raw/u.item', sep='|', names=item_columns, encoding='latin-1')

    # Aggregate genres
    movies['genres'] = movies.apply(aggregate_genres, axis=1)

    # Merge and prepare data
    data_merged = pd.merge(ratings_file, users, on='user_id')
    data_merged = pd.merge(data_merged, movies[['movie_id', 'genres']], on='movie_id')

    # One-Hot Encoding and Normalize age
    data_merged = data_merged.join(data_merged['genres'].str.get_dummies('|'))
    data_merged = pd.get_dummies(data_merged, columns=['gender', 'occupation'])
    scaler = MinMaxScaler()
    data_merged['age_normalized'] = scaler.fit_transform(data_merged['age'].values.reshape(-1, 1))

    # Prepare features for prediction
    X = data_merged.drop(['rating', 'timestamp', 'zip_code', 'age', 'genres'], axis=1)

    # Fit the model
    model = LinearRegression()
    model.fit(X.drop(['user_id', 'movie_id'], axis=1), data_merged['rating'])

    # Prepare user features for prediction
    user_features = X[X['user_id'] == user_id].drop(['user_id', 'movie_id'], axis=1).drop_duplicates()

    if user_features.empty:
        return "User not found in the data."

    # Predict ratings for all movies
    all_movie_ids = movies['movie_id'].unique()
    user_predictions = []

    for movie_id in all_movie_ids:
        movie_features = X[X['movie_id'] == movie_id].drop(['user_id', 'movie_id'], axis=1).iloc[0]
        # Use pandas.concat instead of append
        combined_features = pd.concat([user_features, movie_features.to_frame().T], ignore_index=True)
        predicted_rating = model.predict(combined_features)[-1]
        user_predictions.append((movie_id, predicted_rating))

    # Convert user predictions to a DataFrame
    user_predictions_df = pd.DataFrame(user_predictions, columns=['movie_id', 'predicted_rating'])

    # Filter out movies the user has already rated
    user_interactions = ratings_file[ratings_file['user_id'] == user_id]['movie_id']
    filtered_predictions = user_predictions_df[~user_predictions_df['movie_id'].isin(user_interactions)]

    # Get top K movie IDs with the highest predicted rating
    top_k_movies = filtered_predictions.sort_values(by='predicted_rating', ascending=False).head(k)['movie_id']

    # Map movie IDs to titles
    top_k_movie_titles = movie_titles[movie_titles['movie_id'].isin(top_k_movies)]

    print("Top {} movie recommendations for user {}:".format(k, user_id))
    for id, title in zip(top_k_movie_titles['movie_id'], top_k_movie_titles['title']):
        print("Movie ID: {}, Title: {}".format(id, title))

    return top_k_movie_titles
