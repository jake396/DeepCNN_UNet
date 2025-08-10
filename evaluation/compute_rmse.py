import numpy as np

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Assuming `predicted_output` and `ground_truth_test` have shape (6, 256, 256, 1)
for i in range(6):
    rmse = compute_rmse(ground_truth_test[i], predicted_output[i])
    print(f"Image {i+1} → RMSE: {rmse:.5f}")
