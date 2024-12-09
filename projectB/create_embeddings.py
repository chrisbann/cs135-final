import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from train_valid_test_loader import load_train_valid_test_datasets

def create_embeddings():
    # Load the data (ratings data and metadata)
    train_tuple, valid_tuple, test_tuple, all_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

    movie_info = pd.read_csv('data_movie_lens_100k/movie_info.csv', names=["item_id", "title", "release_year", "orig_item_id"], header = 0)
    user_info = pd.read_csv('data_movie_lens_100k/user_info.csv', names=["user_id", "age", "is_male", "orig_user_id"], header = 0)
    print(movie_info)
    print(user_info)
    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")
    print(len(movie_info))

    # Convert 'release_year' to numeric, forcing errors to NaN
    movie_info['release_year'] = pd.to_numeric(movie_info['release_year'], errors='coerce')
    user_info['age'] = pd.to_numeric(user_info['age'], errors='coerce')

    # Handle missing values in 'release_year' (replace NaN with the median of the column)
    movie_info['release_year'].fillna(movie_info['release_year'].median(), inplace=True)

    # Normalize numerical features
    scaler = MinMaxScaler()
    movie_info['release_year_normalized'] = scaler.fit_transform(movie_info[['release_year']])
    user_info['age_normalized'] = scaler.fit_transform(user_info[['age']])

    # One-hot encode categorical features
    onehot_encoder = OneHotEncoder(sparse=False)
    user_info['is_male'] = onehot_encoder.fit_transform(user_info[['is_male']])

    # Prepare the user and item metadata
    user_metadata_features = user_info[['age_normalized', 'is_male']].values
    item_metadata_features = movie_info[['release_year_normalized']].values

    # Load the ratings matrix for SVD
    ratings_matrix = np.zeros((n_users, n_items))
    for u, i, r in zip(all_tuple[0], all_tuple[1], all_tuple[2]):
        ratings_matrix[int(u), int(i)] = r

    # Apply TruncatedSVD to generate user and item embeddings
    svd = TruncatedSVD(n_components=50)  # Adjust n_components as needed
    user_latent = svd.fit_transform(ratings_matrix)  # User latent embeddings
    item_latent = svd.components_.T  # Item latent embeddings (transpose of components)

    # Combine user and item embeddings with their respective metadata
    enhanced_user_embeddings = np.hstack([user_latent, user_metadata_features])
    enhanced_item_embeddings = np.hstack([item_latent, item_metadata_features])
    print(enhanced_item_embeddings.shape)
    print(enhanced_user_embeddings.shape)
    # Save the enhanced embeddings
    np.save("embeddings/enhanced_user_embeddings.npy", enhanced_user_embeddings)
    np.save("embeddings/enhanced_item_embeddings.npy", enhanced_item_embeddings)

# Run the function to create embeddings
create_embeddings()
