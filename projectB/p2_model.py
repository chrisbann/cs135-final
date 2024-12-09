import numpy as np
from train_valid_test_loader import load_train_valid_test_datasets
from sklearn.metrics import mean_squared_error
# Load datasets
train_tuple, valid_tuple, test_tuple, all_tuple, n_users, n_items = load_train_valid_test_datasets()

# Load embeddings
user_embeddings = np.load("embeddings/enhanced_user_embeddings.npy")  # Shape: (n_users, embedding_dim)
item_embeddings = np.load("embeddings/enhanced_item_embeddings.npy")  # Shape: (n_items, embedding_dim)

def prepare_features_labels(data_tuple, user_embeddings, item_embeddings):
    """
    Prepare features and labels from data tuple.
    
    Args:
        data_tuple (tuple): A tuple containing (user_ids, item_ids, ratings).
        user_embeddings (np.ndarray): Array of user embeddings.
        item_embeddings (np.ndarray): Array of item embeddings.
    
    Returns:
        X (np.ndarray): Combined user and item features for training.
        y (np.ndarray): Corresponding ratings.
    """
    user_ids, item_ids, ratings = data_tuple
    
    # Create features by combining user and item embeddings
    X = np.hstack([user_embeddings[user_ids], item_embeddings[item_ids]])
    y = ratings
    
    return X, y

# Prepare training, validation, and test data
X_train, y_train = prepare_features_labels(train_tuple, user_embeddings, item_embeddings)
X_valid, y_valid = prepare_features_labels(valid_tuple, user_embeddings, item_embeddings)
X_test, y_test = prepare_features_labels(test_tuple, user_embeddings, item_embeddings)
