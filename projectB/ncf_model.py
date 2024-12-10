import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NCFModel(tf.keras.Model):
    def __init__(self, user_embedding_dim, item_embedding_dim, dense_units):
        super(NCFModel, self).__init__()
        # Define user and item dense layers
        self.user_dense = tf.keras.layers.Dense(user_embedding_dim, activation='relu')
        self.item_dense = tf.keras.layers.Dense(item_embedding_dim, activation='relu')
        # Fully connected layers for interaction
        self.fc1 = tf.keras.layers.Dense(dense_units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(dense_units // 2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')  # Rating prediction

        # Save constructor arguments for serialization
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.dense_units = dense_units

    def call(self, inputs):
        user_input, item_input = inputs
        user_features = self.user_dense(user_input)
        item_features = self.item_dense(item_input)
        combined = tf.concat([user_features, item_features], axis=1)
        x = self.fc1(combined)
        x = self.fc2(x)
        return self.output_layer(x)

    def get_config(self):
        # Serialize constructor arguments
        return {
            "user_embedding_dim": self.user_embedding_dim,
            "item_embedding_dim": self.item_embedding_dim,
            "dense_units": self.dense_units,
        }

    @classmethod
    def from_config(cls, config):
        # Deserialize the model from its config
        return cls(**config)



def prepare_features_labels(data_tuple, user_embeddings, item_embeddings):
    user_ids, item_ids, ratings = data_tuple

    # Map user and item IDs to embeddings
    user_features = user_embeddings[user_ids]  # Shape: [num_samples, user_embedding_dim]
    item_features = item_embeddings[item_ids]  # Shape: [num_samples, item_embedding_dim]

    # Ensure user_features and item_features are 2D
    assert user_features.ndim == 2, f"user_features is not 2D: {user_features.shape}"
    assert item_features.ndim == 2, f"item_features is not 2D: {item_features.shape}"
    assert ratings.ndim == 1, f"ratings is not 1D: {ratings.shape}"

    print(f"prepare_features_labels: user_features.shape={user_features.shape}, item_features.shape={item_features.shape}, ratings.shape={ratings.shape}")
    return (user_features, item_features), ratings


def train_ncf_model(model, train_data, valid_data, epochs=10, batch_size=32):
    # Unpack training and validation data
    (user_train, item_train), y_train = train_data
    (user_valid, item_valid), y_valid = valid_data

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(
        [user_train, item_train],  # Provide user and item inputs as a list
        y_train,
        validation_data=([user_valid, item_valid], y_valid),
        epochs=epochs,
        batch_size=batch_size
    )


def evaluate_ncf_model(model, test_data):
    (user_test, item_test), test_labels = test_data

    # Predict using the model
    predictions = model.predict([user_test, item_test])

    # Compute metrics
    mse = mean_squared_error(test_labels, predictions)
    mae = mean_absolute_error(test_labels, predictions)
    return mse, mae
