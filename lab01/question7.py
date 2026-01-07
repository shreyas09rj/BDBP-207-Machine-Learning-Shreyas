#!/usr/bin/python



import numpy as np

X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

theta = np.array([
    [2],
    [3],
    [3]
])

y = np.dot(X, theta)
print(y)



