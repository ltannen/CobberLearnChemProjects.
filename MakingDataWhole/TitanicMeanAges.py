import seaborn as sns
import pandas as pd

# Load Titanic dataset from seaborn
titanic = sns.load_dataset("titanic")

print("=== First 10 rows BEFORE filling missing ages ===")
print(titanic.head(10))   # first 10 rows

print("\nDataset shape:", titanic.shape)
print("\nColumn names:", titanic.columns.tolist())
print("\nSummary info:")
print(titanic.info())

# Calculate mean age (ignores NaN automatically)
mean_age = titanic["age"].mean()
print(f"\nMean age of known values: {mean_age:.2f}")

# Replace null ages with the mean age
titanic["age"] = titanic["age"].fillna(mean_age)

# Verify that missing ages are filled
missing_after = titanic["age"].isna().sum()
print(f"\nNumber of missing ages after filling: {missing_after}")

print("\n=== First 10 rows AFTER filling missing ages ===")
print(titanic.head(10))
