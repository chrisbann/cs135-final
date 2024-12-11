import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ncf_model import NCFModel  # Ensure this is correctly implemented

# Load data and embeddings
user_embeddings = np.load("embeddings/enhanced_user_embeddings.npy")
item_embeddings = np.load("embeddings/enhanced_item_embeddings.npy")
leaderboard_data = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")
dev_data = pd.read_csv("data_movie_lens_100k/ratings_all_development_set.csv")

# Prepare development set
dev_user_ids = dev_data['user_id'].values
dev_item_ids = dev_data['item_id'].values
dev_ratings = dev_data['rating'].values
dev_user_features = user_embeddings[dev_user_ids]
dev_item_features = item_embeddings[dev_item_ids]
dev_features = [dev_user_features, dev_item_features]

# Load the trained model
model = load_model("ncf_model.keras", custom_objects={"NCFModel": NCFModel})

# Predict on the development set
predictions = model.predict(dev_features).flatten()
errors = np.abs(predictions - dev_ratings)

# Add errors and prediction back to the dev_data DataFrame for analysis
dev_data['predicted_rating'] = predictions
dev_data['error'] = errors

# Count number of reviews per movie (item_id)
review_counts = dev_data.groupby('item_id').size().reset_index(name='review_count')

# Merge review counts with dev_data
dev_data = dev_data.merge(review_counts, on='item_id', how='left')

# Group by number of reviews and calculate mean error
review_error_analysis = dev_data.groupby('review_count')['error'].mean().reset_index().rename(columns={'error': 'mean_error'})

# Sort and analyze
high_review_movies = review_error_analysis.sort_values(by='review_count', ascending=False).head(10)
low_review_movies = review_error_analysis.sort_values(by='review_count').head(10)

# Visualization: Mean Error by Review Count
plt.figure(figsize=(12, 6))
plt.scatter(review_error_analysis['review_count'], review_error_analysis['mean_error'], alpha=0.5)
plt.title("Mean Error vs. Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Mean Error")
plt.show()

# Save results for further inspection
high_review_movies.to_csv("high_review_movies.csv", index=False)
low_review_movies.to_csv("low_review_movies.csv", index=False)
review_error_analysis.to_csv("review_error_analysis.csv", index=False)

print("Error analysis by number of reviews complete. Results saved to CSV.")
