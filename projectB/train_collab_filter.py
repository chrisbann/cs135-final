from train_valid_test_loader import load_train_valid_test_datasets
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem

# Load the dataset
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# Set up different values for K
k_values = [2, 10, 50]

# Iterate through the values of K
for k in k_values:
    print(f"\n================================ Training for K = {k} ====================================")
    
    # Initialize the model with no regularization (alpha = 0)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=k, alpha=0.0
    )
    
    # Initialize model parameters
    model.init_parameter_dict(n_users, n_items, train_tuple)
    
    # Train the model using SGD
    model.fit(train_tuple, valid_tuple)
    
    # Evaluate on the validation set
    valid_loss = model.calc_loss_wrt_parameter_dict(model.param_dict, valid_tuple)
    print(f"Validation Loss for K = {k}: {valid_loss:.4f}")
