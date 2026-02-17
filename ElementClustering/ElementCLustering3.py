import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io

# ==========================================
# 1. DATA PREPARATION
# ==========================================
csv_data = """Element Name,Symbol,Atomic Number,Atomic Radius (pm),First Ionization Energy (kJ/mol)
Hydrogen,H,1,53,1312.0
Lithium,Li,3,152,520.2
Sodium,Na,11,186,495.8
Potassium,K,19,227,418.8
Rubidium,Rb,37,248,403.0
Caesium,Cs,55,265,375.7
Francium,Fr,87,348,392.8"""

df = pd.read_csv(io.StringIO(csv_data))

print("--- Interactive Group 1 Clustering Tool ---")

# ==========================================
# 2. SELECT FEATURE
# ==========================================
print("Choose a feature for the X-axis:")
print("1. Atomic Radius (Chemical Property)")
print("2. Atomic Number (Periodic Order)")

while True:
    choice = input("Enter 1 or 2: ")
    if choice == '1':
        x_col = 'Atomic Radius (pm)'
        print(f"\nSelected: {x_col}")
        break
    elif choice == '2':
        x_col = 'Atomic Number'
        print(f"\nSelected: {x_col}")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")

# Define the data for clustering based on choice
X = df[[x_col, 'First Ionization Energy (kJ/mol)']]

print("\nNow, enter a number for 'k' to group the elements.")
print("Enter '0' or 'q' to quit.\n")

# ==========================================
# 3. INTERACTIVE LOOP
# ==========================================
while True:
    user_input = input(f"Enter number of clusters (k) for {x_col}: ")

    # Exit condition
    if user_input.lower() in ['0', 'q', 'quit', 'exit']:
        print("Exiting...")
        break

    # Validate input is an integer
    try:
        k = int(user_input)
        if k < 1 or k > len(df):
            print(f"Please enter a k between 1 and {len(df)}.")
            continue
    except ValueError:
        print("Invalid input. Please enter an integer.")
        continue

    # ==========================================
    # 4. CLUSTERING LOGIC
    # ==========================================
    # Run KMeans with the chosen k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    # Get the coordinates of the cluster centers
    centers = kmeans.cluster_centers_

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Plot the points, colored by cluster
    plt.scatter(df[x_col],
                df['First Ionization Energy (kJ/mol)'],
                c=df['Cluster'],
                cmap='viridis',
                s=100,
                alpha=0.7,
                label='Elements')

    # Plot the Cluster Centers (Large Black X)
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=200, marker='x', linewidths=3, label='Centroids')

    # Add labels to points
    for i, row in df.iterrows():
        plt.annotate(row['Symbol'],
                     (row[x_col], row['First Ionization Energy (kJ/mol)']),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel(x_col)
    plt.ylabel('First Ionization Energy (kJ/mol)')
    plt.title(f'K-Means Clustering: {x_col} vs Ionization Energy (k={k})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Show the plot
    plt.show()

    print(f"Plot generated. Close window to continue.")