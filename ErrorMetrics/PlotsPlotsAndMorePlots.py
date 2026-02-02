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

print("Residuals array:", residuals_array)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R squared (R²):", r2)

# --- PLOT 1: Predicted vs Actual ---
plt.figure(figsize=(6,6))
plt.scatter(actual_array, predicted_array, color='blue', label='Data points')
plt.plot([min(actual_array), max(actual_array)], [min(actual_array), max(actual_array)], 'r--', label='Perfect fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual')
plt.legend()
plt.grid(True)

# Annotate metrics on the plot
metrics_text = f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nR² = {r2:.2f}"
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('predicted_vs_actual_with_metrics.png')
plt.close()

# --- PLOT 2: Residuals ---
plt.figure(figsize=(6,4))
plt.scatter(actual_array, residuals_array, color='green')
plt.axhline(y=0, color='r', linestyle='--')  # horizontal line at 0
plt.xlabel('Actual Values')
plt.ylabel('Residuals (Predicted - Actual)')
plt.title('Residuals Plot')
plt.grid(True)

# Annotate metrics on the residuals plot too
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('residuals_plot_with_metrics.png')
plt.close()

print("Enhanced plots saved as 'predicted_vs_actual_with_metrics.png' and 'residuals_plot_with_metrics.png'.")
