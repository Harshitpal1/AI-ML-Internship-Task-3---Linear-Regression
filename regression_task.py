# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Using the California Housing dataset as a modern alternative to the Boston dataset
from sklearn.datasets import fetch_california_housing

# --- Data Import and Preprocessing --- [cite: 8]
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target # The target variable

# --- Part A: Simple Linear Regression ---
# Objective: Predict MedHouseVal using a single feature (MedInc - Median Income)

print("--- Simple Linear Regression ---")
# Define features (X) and target (y)
X_simple = data[['MedInc']]
y_simple = data['MedHouseVal']

# 2. Split data into training and testing sets [cite: 9]
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# 3. Fit a Linear Regression model [cite: 10]
simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)

# 4. Evaluate the model using MAE, MSE, and R-squared [cite: 11]
y_pred_simple = simple_model.predict(X_test_simple)
mae_simple = metrics.mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = metrics.mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = metrics.r2_score(y_test_simple, y_pred_simple)

print(f"Mean Absolute Error (MAE): {mae_simple:.2f}")
print(f"Mean Squared Error (MSE): {mse_simple:.2f}")
print(f"R-squared (R²): {r2_simple:.2f}")

# 5. Plot regression line and interpret coefficients [cite: 12]
print(f"\nCoefficient (Slope): {simple_model.coef_[0]:.2f}")
print(f"Intercept: {simple_model.intercept_:.2f}")
print("\nInterpretation: For each one-unit increase in Median Income, the predicted Median House Value increases by approximately $0.42 (or $42,000, since the target is in 100,000s).")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test_simple, alpha=0.5, label='Actual Values')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=3, label='Regression Line')
plt.title('Simple Linear Regression: Median Income vs. House Value')
plt.xlabel('Median Income (in tens of thousands)')
plt.ylabel('Median House Value (in $100,000s)')
plt.legend()
plt.grid(True)
plt.show()


# --- Part B: Multiple Linear Regression ---
# Objective: Predict MedHouseVal using all available features

print("\n\n--- Multiple Linear Regression ---")
# Define features (X) and target (y)
X_multi = data.drop('MedHouseVal', axis=1)
y_multi = data['MedHouseVal']

# 2. Split data into training and testing sets [cite: 9]
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# 3. Fit a Linear Regression model [cite: 10]
multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)

# 4. Evaluate the model [cite: 11]
y_pred_multi = multi_model.predict(X_test_multi)
mae_multi = metrics.mean_absolute_error(y_test_multi, y_pred_multi)
mse_multi = metrics.mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = metrics.r2_score(y_test_multi, y_pred_multi)

print(f"Mean Absolute Error (MAE): {mae_multi:.2f}")
print(f"Mean Squared Error (MSE): {mse_multi:.2f}")
print(f"R-squared (R²): {r2_multi:.2f}")

# 5. Interpret coefficients [cite: 12]
coefficients = pd.DataFrame(multi_model.coef_, X_multi.columns, columns=['Coefficient'])
print("\n--- Model Coefficients ---")
print(coefficients)
print("\nInterpretation: Each coefficient represents the change in Median House Value for a one-unit increase in that feature, holding all other features constant.")
