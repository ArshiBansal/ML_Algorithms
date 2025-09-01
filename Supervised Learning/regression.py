# All Regression Models in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# External libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# -----------------------
# Dataset
# -----------------------
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.squeeze()**2 + 2 * X.squeeze() + np.random.randn(100) * 5  # Quadratic with noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for models sensitive to feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Models
# -----------------------
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (deg=2)": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Support Vector Regression (SVR)": SVR(kernel='rbf'),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost": CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
    "Bayesian Regression": BayesianRidge(),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5)
}

# -----------------------
# Training & Evaluation
# -----------------------
results = []
predictions = {}

for name, model in models.items():
    if "Polynomial" in name:
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    elif name in ["Support Vector Regression (SVR)", "Bayesian Regression", "KNN Regression",
                  "Ridge Regression", "Lasso Regression", "ElasticNet Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, r2))
    predictions[name] = y_pred

# -----------------------
# Results Table
# -----------------------
results_df = pd.DataFrame(results, columns=["Model", "MSE", "R² Score"])
print("\nModel Comparison:\n")
print(results_df.sort_values(by="R² Score", ascending=False))

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color="black", label="True Values")

# Plot a few best models
for name in results_df.sort_values(by="R² Score", ascending=False).head(3)["Model"]:
    plt.scatter(X_test, predictions[name], alpha=0.6, label=name)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regression Model Predictions")
plt.legend()
plt.show()
