import numpy as np
from p2_model import train_tuple, valid_tuple, test_tuple
from ncf_model import NCFModel, train_ncf_model, evaluate_ncf_model
from ncf_model import prepare_features_labels  # Ensure this function is correctly implemented

# Load embeddings
user_embeddings = np.load("embeddings/enhanced_user_embeddings.npy")
item_embeddings = np.load("embeddings/enhanced_item_embeddings.npy")

# Debugging: Check shapes of user and item embeddings
print("Debug: Checking Embedding Shapes")
print("User Embeddings Shape:", user_embeddings.shape)
print("Item Embeddings Shape:", item_embeddings.shape)

# Prepare training, validation, and test data
train_data = prepare_features_labels(train_tuple, user_embeddings, item_embeddings)
valid_data = prepare_features_labels(valid_tuple, user_embeddings, item_embeddings)
test_data = prepare_features_labels(test_tuple, user_embeddings, item_embeddings)

# Debugging: Check shapes of prepared data
print("Debug: Checking train_data structure")
print("train_data[0][0].shape (user_features):", train_data[0][0].shape)
print("train_data[0][1].shape (item_features):", train_data[0][1].shape)
print("train_data[1].shape (ratings):", train_data[1].shape)

print("Debug: Checking valid_data structure")
print("valid_data[0][0].shape (user_features):", valid_data[0][0].shape)
print("valid_data[0][1].shape (item_features):", valid_data[0][1].shape)
print("valid_data[1].shape (ratings):", valid_data[1].shape)

# Extract embedding dimensions
user_embedding_dim = user_embeddings.shape[1]
item_embedding_dim = item_embeddings.shape[1]
dense_units = 128

# Define the NCF model
ncf_model = NCFModel(user_embedding_dim, item_embedding_dim, dense_units)

# Train the model
print("Training NCF Model...")
train_ncf_model(
    ncf_model,
    train_data=train_data,  # (user_features, item_features), ratings
    valid_data=valid_data,  # (user_features, item_features), ratings
    epochs=10,
    batch_size=32
)

# Evaluate the model on the test set
print("Evaluating NCF Model...")
test_mse, test_mae = evaluate_ncf_model(ncf_model, test_data)
print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")


# Save in the SavedModel format
ncf_model.save("ncf_model.keras")
print("Model saved in SavedModel format.")
