from train_valid_test_loader import load_train_valid_test_datasets  # Import function to load datasets
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem  # Import collaborative filtering model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()  # Load datasets and metadata
matplotlib.use('Agg')
# Define the values of K to be tested
k_values = [2, 10, 50]  # List of K values to test
alphas = [0.0, 1000]  # List of alpha values to test
results = []  # Initialize list to store results

def generate_plot(train_mae, valid_mae, epoch, title, fname):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch, train_mae, label='Train MAE', marker='o')
    plt.plot(epoch, valid_mae, label='Valid MAE', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{fname}.png')



for k in k_values:  # Iterate over each K value
    for alpha in (alphas if k == 50 else [0.0]):  # Iterate over alpha values, only use 0.1 if K is 50
        print(f"\nTraining model with K = {k}, alpha = {alpha}")  # Print current K and alpha values

        model = CollabFilterOneVectorPerItem(  # Initialize model with specified parameters
            n_epochs=50, batch_size=1000, step_size=0.6,
            n_factors=k, alpha=alpha
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)  # Initialize model parameters
        model.fit(train_tuple, valid_tuple)  # Train model on training data and validate on validation data

        valid_loss = model.evaluate_perf_metrics(*valid_tuple)['mae']  # Calculate validation loss
        test_loss = model.evaluate_perf_metrics(*test_tuple)['mae'] # Calculate test loss

        results.append({  # Append results to the list
            'K': k,
            'alpha': alpha,
            'valid_MAE': valid_loss,
            'test_MAE': test_loss,
            'model': model
        })

        print(f"Validation MAE for K = {k}, alpha = {alpha}: {valid_loss}")  # Print validation loss
        print(f"Test MAE for K = {k}, alpha = {alpha}: {test_loss}")  # Print test loss


for res in results:  # Iterate over results
    print(f"\nK = {res['K']}, alpha = {res['alpha']}")  # Print K and alpha values
    print(f"Validation MAE: {res['valid_MAE']}")  # Print validation loss
    print(f"Test MAE: {res['test_MAE']}")  # Print test loss

    generate_plot(res['model'].trace_mae_train, res['model'].trace_mae_valid, res['model'].trace_epoch , f"Epoch vs MAE for k = " + str(res['K']), str(res['K']))  # Generate plot
