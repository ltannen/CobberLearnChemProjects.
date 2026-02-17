import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Split Data: 75% for training, 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==========================================
# 2. MODEL TRAINING
# ==========================================
# Initialize and train the classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set for evaluation
y_pred = clf.predict(X_test)

# ==========================================
# 3. VISUALIZATION 1: CONFUSION MATRIX
# ==========================================
# Create the matrix comparison
cm = confusion_matrix(y_test, y_pred)

# Display the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Insoluble', 'Soluble'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Actual vs. Predicted")
plt.show()

# ==========================================
# 4. VISUALIZATION 2: DECISION TREE
# ==========================================
# Plot the actual logic tree
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['Insoluble', 'Soluble'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Classifier Logic")
plt.show()