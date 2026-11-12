import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("AirfoilSelfNoise.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))
print("Column names:", df.columns.tolist())

df.columns = df.columns.str.strip()

df["log_f"] = np.log10(df["f"])

print("\nAvailable wind speeds:")
print(sorted(df["U_infinity"].unique()))

# Train using the three slower speeds
# Test using the fastest speed (71.3 m/s)

train = df[df["U_infinity"] < 71]
test = df[df["U_infinity"] > 71]

print("\nTraining rows:", len(train))
print("Testing rows:", len(test))

if len(train) == 0 or len(test) == 0:
    raise ValueError("Check the wind speed values!")

inputs = ["log_f", "alpha", "c", "U_infinity", "delta"]

X_train = train[inputs]
y_train = train["SSPL"]

X_test = test[inputs]
y_test = test["SSPL"]

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test)

print("\nPROJECT RESULTS")

print("R2 Score:",
      round(r2_score(y_test, y_pred), 3))

print("Average error:",
      round(mean_absolute_error(y_test, y_pred), 2), "dB")

print("Average real noise:",
      round(y_test.mean(), 2), "dB")

print("Average predicted noise:",
      round(y_pred.mean(), 2), "dB")

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r-",
    label="Perfect prediction"
)

plt.xlabel("Actual noise at 71.3 m/s (dB)")
plt.ylabel("Predicted noise (dB)")

plt.title("Airfoil Noise Prediction")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("airfoil_noise_prediction.png", dpi=300)
print("Saved graph to airfoil_noise_prediction.png")
plt.show()
