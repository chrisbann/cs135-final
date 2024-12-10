import numpy as np
import pandas as pd
import zipfile
from tensorflow.keras.models import load_model
from ncf_model import NCFModel

# Load the SavedModel format
saved_model = load_model("ncf_model.keras")
print("Loaded SavedModel format successfully.")


# Load embeddings
user_embeddings = np.load("embeddings/enhanced_user_embeddings.npy")
item_embeddings = np.load("embeddings/enhanced_item_embeddings.npy")

# Load leaderboard dataset
leaderboard_data = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")
user_ids = leaderboard_data['user_id'].values  # Extract user IDs
item_ids = leaderboard_data['item_id'].values  # Extract item IDs
print(f"Loaded leaderboard dataset with {len(user_ids)} entries.")

# Prepare leaderboard features
user_features = user_embeddings[user_ids]  # Map user IDs to embeddings
item_features = item_embeddings[item_ids]  # Map item IDs to embeddings
leaderboard_features = [user_features, item_features]  # Prepare features for prediction

# Predict ratings for leaderboard dataset
predicted_ratings = saved_model.predict(leaderboard_features)
predicted_ratings = predicted_ratings.flatten()  # Ensure 1D array of floats
print(f"Predicted Ratings Shape: {predicted_ratings.shape}")

# Save predictions to a plain text file
output_file = "predicted_ratings_leaderboard.txt"
np.savetxt(output_file, predicted_ratings, fmt="%.6f")
print(f"Predictions saved to {output_file}")

# Verify predictions can be reloaded and meet shape requirements
reloaded_predictions = np.loadtxt(output_file)
assert reloaded_predictions.shape == (10000,), "Predictions shape is incorrect!"
print("Predictions file is valid and ready for submission.")

# Create ZIP file for submission
zip_filename = "submission.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipf.write(output_file)
print(f"Submission ZIP file created: {zip_filename}")
