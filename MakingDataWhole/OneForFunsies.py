import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------
# 1️⃣ Load Titanic dataset
# -------------------------------
titanic = sns.load_dataset("titanic")

# -------------------------------
# 2️⃣ Handle missing categorical values safely
# -------------------------------
categorical_cols = titanic.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()

for col in categorical_cols:
    if pd.api.types.is_categorical_dtype(titanic[col]):
        # Add placeholder category first
        if 'Unknown' not in titanic[col].cat.categories:
            titanic[col] = titanic[col].cat.add_categories(['Unknown'])
        titanic[col] = titanic[col].fillna('Unknown')
    else:
        titanic[col] = titanic[col].fillna('missing')

# -------------------------------
# 3️⃣ Prepare target and features
# -------------------------------
y = titanic['age']
X = titanic.drop(columns=['age'])

# One-hot encode categorical and boolean features
X_encoded = pd.get_dummies(X, drop_first=True)

# Impute numeric columns using KNN
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

# -------------------------------
# 4️⃣ Train/test split (known ages)
# -------------------------------
known_mask = ~y.isna()
X_known = X_imputed[known_mask]
y_known = y[known_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_known, y_known, test_size=0.2, random_state=42
)

# -------------------------------
# 5️⃣ Define models
# -------------------------------
models = {
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

# -------------------------------
# 6️⃣ Plot predictions side by side
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plt.subplots_adjust(wspace=0.3)

for ax, (name, model) in zip(axes, models.items()):
    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, color='green')
    ax.plot([0, 80], [0, 80], color='red', linestyle='--')  # perfect prediction line
    ax.set_xlabel("Actual Age")
    ax.set_ylabel("Predicted Age")
    ax.set_title(name)

    # Compute MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Display MAE on the plot
    ax.text(0.05, 0.95, f"MAE: {mae:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))

plt.suptitle("Titanic Age Prediction: KNN vs Random Forest vs Linear Regression", fontsize=16)
plt.show()

# -------------------------------
# 7️⃣ Save predictions for all passengers
# -------------------------------
titanic['age_pred_knn'] = models['KNN'].predict(X_imputed)
titanic['age_pred_rf'] = models['Random Forest'].predict(X_imputed)
titanic['age_pred_linear'] = models['Linear Regression'].predict(X_imputed)

titanic.to_csv("titanic_age_predictions_all_models.csv", index=False)
print("Predictions saved to 'titanic_age_predictions_all_models.csv'.")
