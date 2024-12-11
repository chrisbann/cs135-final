from surprise import Dataset, SVD, Reader
from surprise.model_selection import GridSearchCV
import pandas as pd
import os
import numpy as np

# Load the dataset (only for predictions)
file_path = os.path.expanduser("data_movie_lens_100k/ratings_all_development_set.csv")
file1 = os.path.expanduser("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")

# Read the development and leaderboard datasets into pandas DataFrames
df = pd.read_csv(file_path)
df1 = pd.read_csv(file1)

# Convert 'rating' column from string to float for the development set
df['rating'] = df['rating'].astype(float)

# Define the reader (this is for reading the data into Surprise format)
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

# Create a custom Dataset from the pandas DataFrame for the development set
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Grid search hyperparameter tuning for SVD model
param_grid = {
    "n_epochs": [20, 25],
    "lr_all": [0.004, 0.005],
    "reg_all": [0.4, 0.],
}

# Perform grid search
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)
gs.fit(data)

# Output the best MAE score and corresponding parameters
print(f"Best MAE score: {gs.best_score['mae']}")
print(f"Best parameters for MAE: {gs.best_params['mae']}")

# Retrieve the best model from grid search
best_model = gs.best_estimator['mae']

# Train the best model on the entire development dataset (no validation split, just fit on all data)
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Predict ratings for the held-out validation set (from file1)
heldout_data = pd.read_csv(file1)  # assuming this contains user-item pairs for which you want to predict ratings

# Predict ratings for the held-out data
predicted_ratings = heldout_data.apply(
    lambda row: best_model.predict(row['user_id'], row['item_id']).est, axis=1
)

# Save only the predicted ratings to a file (no user_id or item_id)
np.savetxt("predicted_ratings_leaderboard.txt", predicted_ratings, fmt="%.6f")

# Optionally, print the first few predictions
print(predicted_ratings.head())
