import numpy as np
import matplotlib.pyplot as plt
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

# --- Create combined figure ---
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# Color-coded Predicted vs Actual (left subplot)
over_predicted = residuals_array > 0
under_predicted = residuals_array < 0

axes[0].scatter(actual_array[over_predicted], predicted_array[over_predicted], color='red', label='Over-predicted')
axes[0].scatter(actual_array[under_predicted], predicted_array[under_predicted], color='blue', label='Under-predicted')
axes[0].plot([min(actual_array), max(actual_array)], [min(actual_array), max(actual_array)], 'k--', label='Perfect fit')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Predicted vs Actual')
axes[0].legend()
axes[0].grid(True)
axes[0].text(0.05, 0.95, f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nR² = {r2:.2f}",
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

# Color-coded Residuals plot (right subplot)
axes[1].scatter(actual_array[residuals_array>0], residuals_array[residuals_array>0], color='red', label='Over-predicted')
axes[1].scatter(actual_array[residuals_array<0], residuals_array[residuals_array<0], color='blue', label='Under-predicted')
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Residuals (Predicted - Actual)')
axes[1].set_title('Residuals Plot')
axes[1].legend()
axes[1].grid(True)
axes[1].text(0.05, 0.95, f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nR² = {r2:.2f}",
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('combined_plots.png')
plt.close()

print("Combined figure saved as 'combined_plots.png'.")
