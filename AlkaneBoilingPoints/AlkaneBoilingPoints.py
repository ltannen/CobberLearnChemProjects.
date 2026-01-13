import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons

# Use Georgia font for all text
plt.rcParams['font.family'] = 'Georgia'

# Data
num_carbons = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
boiling_points = np.array([
    -161.5, -88.6, -42.1, -0.5, 36.1,
    68.7, 98.4, 125.6, 150.8, 174.1
])

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.22)

# Scatter plot
ax.scatter(num_carbons, boiling_points, zorder=3)

# Title and labels
title = ax.set_title("Boiling Point vs. Number of Carbons for Linear Alkanes", pad=15)
ax.set_xlabel("Number of Carbon Atoms")
ax.set_ylabel("Boiling Point (°C)")

# Axes formatting
ax.set_xticks(range(0, 12))
ax.set_xlim(0, 11)

# Grid
ax.grid(True, which='both', axis='both',
        color='lightgray', linestyle='--', linewidth=0.5, zorder=0)

# Placeholder for fit line
fit_line, = ax.plot([], [], zorder=2)

# Placeholder for equation text (next to title)
equation_text = ax.text(
    0.99, 1.01, "",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=10
)

# Function to update fit
def update_fit(label):
    if label == "None":
        fit_line.set_data([], [])
        equation_text.set_text("")
    else:
        degree_map = {
            "Linear": 1,
            "Quadratic": 2,
            "Cubic": 3
        }
        degree = degree_map[label]

        coeffs = np.polyfit(num_carbons, boiling_points, degree)
        x_fit = np.linspace(1, 10, 300)
        y_fit = np.polyval(coeffs, x_fit)

        fit_line.set_data(x_fit, y_fit)

        # Format equation text
        if degree == 1:
            a, b = coeffs
            eq = f"y = {a:.2f}x + {b:.2f}"
        elif degree == 2:
            a, b, c = coeffs
            eq = f"y = {a:.2f}x² + {b:.2f}x + {c:.2f}"
        else:
            a, b, c, d = coeffs
            eq = f"y = {a:.2f}x³ + {b:.2f}x² + {c:.2f}x + {d:.2f}"

        equation_text.set_text(eq)

    fig.canvas.draw_idle()

# Radio button placement (right of x-axis label)
rax = plt.axes([0.72, 0.05, 0.25, 0.12])
radio = RadioButtons(
    rax,
    ("None", "Linear", "Quadratic", "Cubic"),
    active=0
)

radio.on_clicked(update_fit)

# Show plot
plt.show()


