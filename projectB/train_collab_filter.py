from train_valid_test_loader import load_train_valid_test_datasets
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem

# Load the dataset
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# Experiment settings
k_values = [2, 10, 50]
alphas = [0.0, 0.1]  # No regularization for first runs, then regularization for K=50

results = []

# Loop through different K values
for k in k_values:
    for alpha in (alphas if k == 50 else [0.0]):  # Regularization only for K=50
        print(f"\nTraining model with K = {k}, alpha = {alpha}")

        # Initialize and train the model
        model = CollabFilterOneVectorPerItem(
            n_epochs=20, batch_size=10000, step_size=0.1,
            n_factors=k, alpha=alpha
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)

        # Evaluate performance
        valid_loss = model.calc_loss_wrt_parameter_dict(model.param_dict, valid_tuple)
        test_loss = model.calc_loss_wrt_parameter_dict(model.param_dict, test_tuple)

        # Record results
        results.append({
            'K': k,
            'alpha': alpha,
            'valid_MAE': valid_loss,
            'test_MAE': test_loss,
            'parameters': model.param_dict
        })

        print(f"Validation MAE for K = {k}, alpha = {alpha}: {valid_loss}")
        print(f"Test MAE for K = {k}, alpha = {alpha}: {test_loss}")

# Summarize results
for res in results:
    print(f"K = {res['K']}, alpha = {res['alpha']}, Valid MAE: {res['valid_MAE']}, Test MAE: {res['test_MAE']}")
