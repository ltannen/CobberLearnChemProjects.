import matplotlib.pyplot as plt

# Number of carbons in the first 10 linear alkanes (methane to decane)
num_carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Corresponding boiling points in degrees Celsius
boiling_points = [
    -161.5,  # Methane
    -88.6,   # Ethane
    -42.1,   # Propane
    -0.5,    # Butane
    36.1,    # Pentane
    68.7,    # Hexane
    98.4,    # Heptane
    125.6,   # Octane
    150.8,   # Nonane
    174.1    # Decane
]

# Create the scatter plot
plt.scatter(num_carbons, boiling_points)

# Add title and axis labels
plt.title("Boiling Point vs. Number of Carbons for Linear Alkanes")
plt.xlabel("Number of Carbon Atoms")
plt.ylabel("Boiling Point (°C)")

# Display the plot
plt.show()
