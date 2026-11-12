
# PROJECT A06
# Can the Model Predict a Faster Wind Speed?

# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# STEP 2: Load the dataset
df = pd.read_csv("AirfoilSelfNoise.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))
print("Column names:", df.columns.tolist())

# STEP 3: Prepare the data
df.columns = df.columns.str.strip()

# Convert frequency into log10
df["log_f"] = np.log10(df["f"])

print("\nAvailable wind speeds:")
print(sorted(df["U_infinity"].unique()))

# STEP 4: Split the data
# Train using the three slower speeds
# Test using the fastest speed (71.3 m/s)

train = df[df["U_infinity"] < 71]
test = df[df["U_infinity"] > 71]

print("\nTraining rows:", len(train))
print("Testing rows:", len(test))

if len(train) == 0 or len(test) == 0:
    raise ValueError("Check the wind speed values!")

# STEP 5: Select inputs and output
inputs = ["log_f", "alpha", "c", "U_infinity", "delta"]

X_train = train[inputs]
y_train = train["SSPL"]

X_test = test[inputs]
y_test = test["SSPL"]

# STEP 6: Train the Machine Learning model
model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed!")

# STEP 7: Predict the noise level
y_pred = model.predict(X_test)

# STEP 8: Evaluate the model
print("\nPROJECT RESULTS")

print("R2 Score:",
      round(r2_score(y_test, y_pred), 3))

print("Average error:",
      round(mean_absolute_error(y_test, y_pred), 2), "dB")

print("Average real noise:",
      round(y_test.mean(), 2), "dB")

print("Average predicted noise:",
      round(y_pred.mean(), 2), "dB")

# STEP 9: Draw the graph
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

plt.show()
