import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data
actual = [2, 4, 5, 4, 5, 7, 9]
predicted = [2.5, 3.5, 4, 5, 6, 8, 8]

# Convert to NumPy arrays
actual_array = np.array(actual)
predicted_array = np.array(predicted)

# Calculate residuals
residuals_array = predicted_array - actual_array

# Calculate metrics
mae = mean_absolute_error(actual_array, predicted_array)
mse = mean_squared_error(actual_array, predicted_array)
r2 = r2_score(actual_array, predicted_array)

# Print results
print("Residuals array:", residuals_array)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R squared (R²):", r2)
