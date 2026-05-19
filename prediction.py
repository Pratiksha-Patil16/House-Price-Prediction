import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/housing.csv")

# Show dataset
print(df.head())

# Convert categorical columns into numeric
df.replace({
    'yes': 1,
    'no': 0,
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
}, inplace=True)

# Convert object columns using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------- Linear Regression ----------------

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLinear Regression")

print("R2 Score:", r2_score(y_test, lr_pred))

print("Mean Absolute Error:", mean_absolute_error(y_test, lr_pred))


# ---------------- Decision Tree ----------------

dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Regressor")

print("R2 Score:", r2_score(y_test, dt_pred))

print("Mean Absolute Error:", mean_absolute_error(y_test, dt_pred))


# ---------------- Random Forest ----------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Regressor")

print("R2 Score:", r2_score(y_test, rf_pred))

print("Mean Absolute Error:", mean_absolute_error(y_test, rf_pred))

# Compare Actual vs Predicted

plt.figure(figsize=(8,5))

plt.scatter(y_test, rf_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()

# Save Linear Regression Model
import pickle
with open("house_model.pkl", "wb") as file:
    pickle.dump(lr_model, file)

print("Model saved successfully")