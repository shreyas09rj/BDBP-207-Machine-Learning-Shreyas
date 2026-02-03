#!/usr/bin/python



# Hypothesis Function

def hypothesis(x, theta):

    return theta[0] + theta[1] * x




def cost_function(X, y, theta):

    m = len(X)
    total = 0
    for i in range(m):
        total += (hypothesis(X[i], theta) - y[i]) ** 2
    return total / (2 * m)


# Gradient Descent

def gradient_descent(X, y, alpha=0.01, epochs=1000):
    theta = [0, 0]
    m = len(X)

    for _ in range(epochs):
        d0 = 0
        d1 = 0

        for i in range(m):
            error = hypothesis(X[i], theta) - y[i]
            d0 += error
            d1 += error * X[i]

        d0 = d0 / m
        d1 = d1 / m

        theta[0] = theta[0] - alpha * d0
        theta[1] = theta[1] - alpha * d1

    return theta



def r2_score(y, y_pred):
    mean_y = sum(y) / len(y)

    ss_tot = 0
    ss_res = 0

    for i in range(len(y)):
        ss_tot += (y[i] - mean_y) ** 2
        ss_res += (y[i] - y_pred[i]) ** 2

    r2 = 1 - (ss_res / ss_tot)
    return r2


# Example Dataset
# y = 2x + 1

X = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# Train model
theta = gradient_descent(X, y, alpha=0.1, epochs=1000)

# Predictions
y_pred = [hypothesis(x, theta) for x in X]

# R2 score
r2 = r2_score(y, y_pred)

print("Learned Parameters:")
print("Theta0 (Intercept):", theta[0])
print("Theta1 (Slope):", theta[1])

print("\nPredicted values:", y_pred)
print("Actual values   :", y)

print("\nR2 Score (from scratch):", r2)

# Compare with Scikit-learn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as sk_r2

X_2d = [[x] for x in X]
model = LinearRegression()
model.fit(X_2d, y)
y_pred_sk = model.predict(X_2d)

print("\nScikit-learn R2:", sk_r2(y, y_pred_sk))


