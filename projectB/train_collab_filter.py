from train_valid_test_loader import load_train_valid_test_datasets  # Import function to load datasets
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem  # Import collaborative filtering model

train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()  # Load datasets and metadata

# Define the values of K to be tested
k_values = [2, 10, 50]  # List of K values to test
alphas = [0.0, 0.1]  # List of alpha values to test

results = []  # Initialize list to store results

for k in k_values:  # Iterate over each K value
    for alpha in (alphas if k == 50 else [0.0]):  # Iterate over alpha values, only use 0.1 if K is 50
        print(f"\nTraining model with K = {k}, alpha = {alpha}")  # Print current K and alpha values

        model = CollabFilterOneVectorPerItem(  # Initialize model with specified parameters
            n_epochs=20, batch_size=10000, step_size=0.1,
            n_factors=k, alpha=alpha
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)  # Initialize model parameters
        model.fit(train_tuple, valid_tuple)  # Train model on training data and validate on validation data

        valid_loss = model.calc_loss_wrt_parameter_dict(model.param_dict, valid_tuple)  # Calculate validation loss
        test_loss = model.calc_loss_wrt_parameter_dict(model.param_dict, test_tuple)  # Calculate test loss

        results.append({  # Append results to the list
            'K': k,
            'alpha': alpha,
            'valid_MAE': valid_loss,
            'test_MAE': test_loss,
            'parameters': model.param_dict
        })

        print(f"Validation MAE for K = {k}, alpha = {alpha}: {valid_loss}")  # Print validation loss
        print(f"Test MAE for K = {k}, alpha = {alpha}: {test_loss}")  # Print test loss

for res in results:  # Iterate over results
    print(f"K = {res['K']}, alpha = {res['alpha']}, Valid MAE: {res['valid_MAE']}, Test MAE: {res['test_MAE']}")  # Print summary of results
