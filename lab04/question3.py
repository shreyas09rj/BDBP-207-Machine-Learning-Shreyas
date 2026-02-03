#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# CHANGE ONLY THESE TWO LINES

filename = "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv"
target_col = "disease_score_fluct"

# Load Data

df = pd.read_csv(filename)
X = df.drop(target_col, axis=1).values
y = df[target_col].values.reshape(-1, 1)

# Add bias column (for Normal Eq)

ones = np.ones((X.shape[0], 1))
X_bias = np.hstack((ones, X))


# 1. NORMAL EQUATION
# θ = (XᵀX)⁻¹ Xᵀ y

Xt = X_bias.T
theta_normal = np.linalg.inv(Xt.dot(X_bias)).dot(Xt).dot(y)
y_pred_normal = X_bias.dot(theta_normal)

ss_res = np.sum((y - y_pred_normal) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_normal = 1 - ss_res / ss_tot


# 2. GRADIENT DESCENT

# Normalize features
X_gd = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_gd = np.hstack((np.ones((X_gd.shape[0], 1)), X_gd))

theta = np.zeros((X_gd.shape[1], 1))
alpha = 0.01
epochs = 1000
m = len(y)

for i in range(epochs):
    y_pred = X_gd.dot(theta)
    gradient = (1/m) * X_gd.T.dot(y_pred - y)
    theta = theta - alpha * gradient

theta_gd = theta
y_pred_gd = X_gd.dot(theta_gd)

ss_res = np.sum((y - y_pred_gd) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_gd = 1 - ss_res / ss_tot


# 3. SCIKIT-LEARN

model = LinearRegression()
model.fit(X, y)
y_pred_sklearn = model.predict(X)
r2_sklearn = r2_score(y, y_pred_sklearn)


# RESULTS

print("R2 (Normal Equation):   ", r2_normal)
print("R2 (Gradient Descent):  ", r2_gd)
print("R2 (scikit-learn):      ", r2_sklearn)

print("\nTheta (Normal Equation):")
print(theta_normal)

print("\nTheta (Gradient Descent):")
print(theta_gd)

print("\nCoefficients (sklearn):")
print("Intercept:", model.intercept_)
print("Weights:", model.coef_)
