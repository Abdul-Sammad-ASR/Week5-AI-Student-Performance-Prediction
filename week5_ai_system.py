import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. LOAD DATASET
# =========================================================
data = pd.read_csv(r"D:\INTERNSHIP\week 5\student_performance.csv")

print("Student Performance Dataset:")
print(data)
print("\nDataset Shape:", data.shape)

# =========================================================
# 2. DATA PREPROCESSING
# =========================================================
print("\nMissing values per column:")
print(data.isnull().sum())

# Remove duplicates and fill any missing values with column mean
data = data.drop_duplicates()
data = data.fillna(data.mean(numeric_only=True))

print("\nStatistical Summary:")
print(data.describe())

# Separate features and target
features = ["Study_Hours", "Attendance", "Previous_Marks", "Sleep_Hours"]
X = data[features]
y = data["Final_Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (fit on training data only to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# =========================================================
# 3. TRAIN MODEL
# =========================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\nModel trained successfully!")

# =========================================================
# 4. ACCURACY EVALUATION
# =========================================================
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"Root Mean Squared Error  : {rmse:.2f}")
print(f"R2 Score (Accuracy)      : {r2:.4f}  ({r2 * 100:.2f}%)")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": np.round(y_pred, 2)
})
print("\nActual vs Predicted:")
print(comparison)

# Feature importance (coefficients on scaled data)
importance = pd.Series(model.coef_, index=features).sort_values(ascending=False)
print("\nFeature Importance (coefficients):")
print(importance)

# =========================================================
# 5. RESULT VISUALIZATION
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Student Performance Prediction System", fontsize=16, fontweight="bold")

# (a) Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, color="royalblue", s=80, edgecolors="black")
lims = [min(y_test.min(), y_pred.min()) - 3, max(y_test.max(), y_pred.max()) + 3]
axes[0, 0].plot(lims, lims, "r--", label="Perfect Prediction")
axes[0, 0].set_xlabel("Actual Marks")
axes[0, 0].set_ylabel("Predicted Marks")
axes[0, 0].set_title("Actual vs Predicted")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# (b) Feature importance
axes[0, 1].barh(importance.index, importance.values, color="seagreen")
axes[0, 1].set_title("Feature Importance")
axes[0, 1].set_xlabel("Coefficient")

# (c) Study hours vs Final marks
axes[1, 0].scatter(data["Study_Hours"], data["Final_Marks"], color="darkorange", edgecolors="black")
z = np.polyfit(data["Study_Hours"], data["Final_Marks"], 1)
xs = np.linspace(data["Study_Hours"].min(), data["Study_Hours"].max(), 50)
axes[1, 0].plot(xs, np.polyval(z, xs), "r-")
axes[1, 0].set_xlabel("Study Hours")
axes[1, 0].set_ylabel("Final Marks")
axes[1, 0].set_title("Study Hours vs Final Marks")
axes[1, 0].grid(alpha=0.3)

# (d) Correlation heatmap
corr = data.corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr.columns)))
axes[1, 1].set_yticks(range(len(corr.columns)))
axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha="right")
axes[1, 1].set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
axes[1, 1].set_title("Correlation Heatmap")
fig.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()

# =========================================================
# 6. USER INPUT + PREDICTION SYSTEM
# =========================================================
def get_number(prompt, low, high):
    """Keep asking until the user enters a valid number in range."""
    while True:
        try:
            value = float(input(prompt))
            if low <= value <= high:
                return value
            print(f"  Please enter a value between {low} and {high}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def predict_marks(study, attendance, prev_marks, sleep):
    new_data = pd.DataFrame([[study, attendance, prev_marks, sleep]], columns=features)
    new_scaled = scaler.transform(new_data)
    prediction = model.predict(new_scaled)[0]
    return float(np.clip(prediction, 0, 100))


def grade(marks):
    if marks >= 85: return "A (Excellent)"
    if marks >= 70: return "B (Good)"
    if marks >= 55: return "C (Average)"
    if marks >= 40: return "D (Needs Improvement)"
    return "F (At Risk)"


print("\n" + "=" * 50)
print("   STUDENT FINAL MARKS PREDICTION SYSTEM")
print("=" * 50)

while True:
    study = get_number("\nStudy hours per day (0-16): ", 0, 16)
    attendance = get_number("Attendance % (0-100): ", 0, 100)
    prev = get_number("Previous marks (0-100): ", 0, 100)
    sleep = get_number("Sleep hours per day (0-12): ", 0, 12)

    result = predict_marks(study, attendance, prev, sleep)
    print("\n--- Prediction Result ---")
    print(f"Predicted Final Marks: {result:.2f}")
    print(f"Predicted Grade      : {grade(result)}")

    # Visualize this prediction against the dataset
    plt.figure(figsize=(8, 5))
    plt.hist(data["Final_Marks"], bins=8, color="lightsteelblue", edgecolor="black", label="Dataset marks")
    plt.axvline(result, color="red", linewidth=2.5, label=f"Your prediction: {result:.1f}")
    plt.xlabel("Final Marks")
    plt.ylabel("Number of Students")
    plt.title("Your Predicted Marks vs Dataset Distribution")
    plt.legend()
    plt.show()

    again = input("Predict for another student? (yes/no): ").strip().lower()
    if again not in ("yes", "y"):
        print("\nThank you for using the system!")
        break
