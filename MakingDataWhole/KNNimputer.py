import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
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
# 2️⃣ Numeric features only
# -------------------------------
numeric_cols = titanic.select_dtypes(include=["int64", "float64"]).columns.tolist()
X = titanic[numeric_cols].drop(columns=["age"])
y = titanic["age"]

# Impute missing feature values with KNN
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/test split on known ages
known_mask = ~y.isna()
X_known = X_imputed[known_mask]
y_known = y[known_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_known, y_known, test_size=0.2, random_state=42
)

# -------------------------------
# 3️⃣ Prepare figure
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.3)  # leave space for radio buttons

# Add a placeholder for MAE text
mae_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))


# -------------------------------
# 4️⃣ Train and plot function
# -------------------------------
def train_and_plot(model_name):
    # Select model
    if model_name == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "linear":
        model = LinearRegression()
    elif model_name == "knn":
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError("Invalid model name")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Clear previous plot
    ax.clear()

    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, color="green")
    ax.plot([0, 80], [0, 80], color="red", linestyle="--")
    ax.set_xlabel("Actual Age")
    ax.set_ylabel("Predicted Age")
    ax.set_title(f"Age Prediction using {model_name.upper()}")

    # Compute MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Update MAE text on figure
    ax.text(0.05, 0.95, f"MAE: {mae:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))

    # Refresh figure
    fig.canvas.draw_idle()


# Initial plot
train_and_plot("rf")

# -------------------------------
# 5️⃣ Radio buttons for model switching
# -------------------------------
axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.05, 0.4, 0.15, 0.2], facecolor=axcolor)
radio = RadioButtons(rax, ('knn', 'rf', 'linear'))


def model_switch(label):
    train_and_plot(label)


radio.on_clicked(model_switch)

plt.show()
