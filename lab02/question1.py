import numpy as np

X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

mean = np.mean(X, axis=0)


X_centered = X - mean

# Covariance using matrix multiplication
cov_manual = (X_centered.T @ X_centered) / X.shape[0]

print("Mean:")
print(mean)

print("\nCovariance (manual):")
print(cov_manual)

cov_numpy = np.cov(X, rowvar=False, bias=True)

print("\nCovariance (NumPy):")
print(cov_numpy)
