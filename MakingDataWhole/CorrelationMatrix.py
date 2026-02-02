import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Encode categorical variables
titanic_encoded = titanic.copy()
categorical_cols = ["sex", "class", "embark_town", "who", "adult_male", "alone"]

for col in categorical_cols:
    titanic_encoded[col] = titanic_encoded[col].astype("category").cat.codes

# Select numeric columns
numeric_data = titanic_encoded.select_dtypes(include="number")

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# -------------------------------
# Save correlation matrix to file
# -------------------------------

# Get current project directory (PyCharm working directory)
project_dir = os.getcwd()
csv_path = os.path.join(project_dir, "titanic_correlation_matrix.csv")

corr_matrix.to_csv(csv_path)

print(f"Correlation matrix saved to:\n{csv_path}")

# -------------------------------
# Save heatmap image
# -------------------------------

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Titanic Dataset Correlation Matrix")

img_path = os.path.join(project_dir, "titanic_correlation_heatmap.png")
plt.savefig(img_path, bbox_inches="tight")
plt.close()

print(f"Correlation heatmap saved to:\n{img_path}")
