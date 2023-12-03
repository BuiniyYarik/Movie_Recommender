import numpy as np


def calculate_hit_ratio_svd(test_data, predicted_ratings_df, all_movies):
    """
    Calculate the hit ratio for the SVD model

    Parameters:
    test_data (pandas.DataFrame): The test data
    predicted_ratings_df (pandas.DataFrame): The predicted ratings
    all_movies (list): The list of all movies

    Returns:
    hit_ratio (float): The hit ratio
    """
    # Create a dictionary of user interactions based on the test data
    user_interacted_items = test_data.groupby('user_id')['movie_id'].apply(set).to_dict()
    hits = []

    for user_id in test_data['user_id'].unique():
        # Get a high-rated item and 99 unseen items for this user
        high_rating_items = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] >= 4)]
        if high_rating_items.empty:
            high_rating_items = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] > 3)]
        if high_rating_items.empty:
            continue

        test_item = high_rating_items.sample(1)['movie_id'].iloc[0]
        interacted_items = user_interacted_items.get(user_id, set())
        not_interacted_items = set(all_movies) - interacted_items
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99, replace=False))
        test_items = selected_not_interacted + [test_item]

        # Check if the test items are in the predicted ratings
        available_ratings = predicted_ratings_df.loc[user_id].reindex(test_items)
        available_ratings = available_ratings.dropna()

        # Get the top 10 predicted items
        top10_items = available_ratings.sort_values(ascending=False).head(10).index.tolist()

        hits.append(int(test_item in top10_items))

    hit_ratio = np.mean(hits)
    return hit_ratio
