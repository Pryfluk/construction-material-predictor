import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("data/construction_training_data.csv")

# -----------------------------
# 2. Define features and target
# -----------------------------
X = data[
    [
        "floor_area",
        "floor_height",
        "column_count",
        "beam_count",
        "slab_thickness"
    ]
]

y = data["concrete_volume"]

# -----------------------------
# 3. Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Model Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results")
print("------------------------")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# -----------------------------
# 7. Cross-validation
# -----------------------------
cv_scores = cross_val_score(
    model, X, y, cv=5, scoring="r2"
)

print("\nCross-validation (R²)")
print("------------------------")
print(f"Scores     : {cv_scores}")
print(f"Mean R²    : {cv_scores.mean():.3f}")
print(f"Std (±)    : {cv_scores.std():.3f}")

# -----------------------------
# 8. Save model
# -----------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/concrete_estimation_model.pkl")

# -----------------------------
# 9. Save evaluation result
# -----------------------------
with open("model/model_evaluation.txt", "w", encoding="utf-8") as f:
    f.write("Model Evaluation Results\n")
    f.write("------------------------\n")
    f.write(f"MAE  : {mae:.2f}\n")
    f.write(f"MSE  : {mse:.2f}\n")
    f.write(f"RMSE : {rmse:.2f}\n")
    f.write(f"R²   : {r2:.3f}\n\n")
    f.write("Cross-validation (R²)\n")
    f.write(f"Scores  : {cv_scores}\n")
    f.write(f"Mean R² : {cv_scores.mean():.3f}\n")
    f.write(f"Std     : {cv_scores.std():.3f}\n")

print("\nModel and evaluation results saved successfully.")