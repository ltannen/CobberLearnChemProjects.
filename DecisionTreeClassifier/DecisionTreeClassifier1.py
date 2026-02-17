import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. DATA PREPARATION
# ==========================================
data = {
    'Molecular Weight': [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    'H-Bond Donors': [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    'H-Bond Acceptors': [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    'Water Solubility': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data, index=[f'Molecule {i}' for i in range(1, 13)])

# Separate Features (X) and Target (y)
X = df.drop('Water Solubility', axis=1)
y = df['Water Solubility']

# Split into Train and Test sets (25% for testing = 3 molecules)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==========================================
# 2. MODEL TRAINING
# ==========================================
# Initialize and fit the Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 3. VISUALIZATION: CONFUSION MATRIX
# ==========================================
# Generate the matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting using Seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Insoluble (0)', 'Predicted Soluble (1)'],
            yticklabels=['Actual Insoluble (0)', 'Actual Soluble (1)'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix Results')
plt.show()

# ==========================================
# 4. VISUALIZATION: DECISION TREE STRUCTURE
# ==========================================
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['Insoluble', 'Soluble'],
          filled=True,
          rounded=True)
plt.title("Decision Tree Logic")
plt.show()