import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("Salary_dataset.csv")

X = df["YearsExperience"].values.reshape(-1, 1)
y = df["Salary"].values.reshape(-1, 1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    verbose=0
)

pred_scaled = model.predict(X_test, verbose=0)

predictions = y_scaler.inverse_transform(pred_scaled)
actuals = y_scaler.inverse_transform(y_test)

r2 = r2_score(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print("\n========== RESULTS ==========")
print(f"MAE : ₹{mae:,.2f}")
print(f"R²  : {r2:.4f}")

weights = model.layers[0].get_weights()

print("\nModel Parameters")
print("Weight:", weights[0][0][0])
print("Bias:", weights[1][0])

plt.figure(figsize=(10, 6))

plt.scatter(
    actuals,
    predictions
)

min_val = min(actuals.min(), predictions.min())
max_val = max(actuals.max(), predictions.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val]
)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))

plt.plot(history.history["loss"])

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)

plt.show()


all_pred_scaled = model.predict(X_scaled, verbose=0)
all_predictions = y_scaler.inverse_transform(all_pred_scaled)

sorted_idx = np.argsort(X.flatten())

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X.flatten()[sorted_idx],
    all_predictions.flatten()[sorted_idx],
    linewidth=2,
    label="Model Prediction"
)

plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.title("Salary Regression")
plt.legend()
plt.grid(True)
plt.show()


while True:

    years = input(
        "\nEnter years of experience (or q to quit): "
    )

    if years.lower() == "q":
        break

    years = float(years)

    years_scaled = x_scaler.transform([[years]])

    pred_scaled = model.predict(
        years_scaled,
        verbose=0
    )

    predicted_salary = y_scaler.inverse_transform(
        pred_scaled
    )[0][0]

    print(
        f"Predicted Salary: {predicted_salary:,.2f}"
    )
