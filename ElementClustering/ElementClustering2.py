import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io

# 1. DATA PREPARATION
csv_data = """Element Name,Symbol,Atomic Number,Atomic Radius (pm),First Ionization Energy (kJ/mol)
Hydrogen,H,1,53,1312.0
Lithium,Li,3,152,520.2
Sodium,Na,11,186,495.8
Potassium,K,19,227,418.8
Rubidium,Rb,37,248,403.0
Caesium,Cs,55,265,375.7
Francium,Fr,87,348,392.8"""

# Load into DataFrame
df = pd.read_csv(io.StringIO(csv_data))

# Select the features we want to use for clustering
X = df[['Atomic Radius (pm)', 'First Ionization Energy (kJ/mol)']]

# 2. PERFORM K-MEANS CLUSTERING
# We ask for 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 3. VISUALIZATION
plt.figure(figsize=(10, 6))

# Define colors for the clusters
colors = ['red', 'blue']

# Plot each cluster separately to create a legend
for cluster_id in range(2):
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Atomic Radius (pm)'],
                cluster_data['First Ionization Energy (kJ/mol)'],
                color=colors[cluster_id],
                label=f'Cluster {cluster_id + 1}',
                s=100, alpha=0.7)

# Add labels for each element
for i, row in df.iterrows():
    plt.annotate(row['Symbol'],
                 (row['Atomic Radius (pm)'], row['First Ionization Energy (kJ/mol)']),
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Atomic Radius (pm)')
plt.ylabel('First Ionization Energy (kJ/mol)')
plt.title('K-Means Clustering of Group 1 Elements (2 Clusters)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()

# Print the grouping results
print(df[['Symbol', 'Cluster']])