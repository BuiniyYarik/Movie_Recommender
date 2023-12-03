import numpy as np


def calculate_hit_ratio_lr(test_data, predicted_ratings, all_movies):
    """
    Calculate the hit ratio for the Linear Regression model

    Parameters:
    test_data (pandas.DataFrame): The test data
    predicted_ratings (pandas.DataFrame): The predicted ratings
    all_movies (list): The list of all movies

    Returns:
    hit_ratio (float): The hit ratio
    """
    hits = []
    for user_id in test_data['user_id'].unique():
        high_rating_items = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] >= 4)]
        if high_rating_items.empty:
            high_rating_items = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] > 3)]
        if high_rating_items.empty:
            continue
        test_item = high_rating_items.sample(1)['movie_id'].iloc[0]
        interacted_items = set(test_data[test_data['user_id'] == user_id]['movie_id'])
        not_interacted_items = set(all_movies) - interacted_items

        # Adjust the number of items to sample based on the available non-interacted items
        num_sample = min(len(not_interacted_items), 99)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), num_sample, replace=False))

        test_items = selected_not_interacted + [test_item]

        # Handle missing movies
        available_ratings = predicted_ratings.loc[(user_id, slice(None)), :].droplevel(0)
        available_ratings = available_ratings[available_ratings.index.isin(test_items)]
        top10_items = available_ratings.sort_values(by='predicted_rating', ascending=False).head(10).index.tolist()

        hits.append(int(test_item in top10_items))
    return np.mean(hits)
