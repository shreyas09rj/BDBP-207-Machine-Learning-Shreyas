#!/usr/bin/python

# Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables. Compute the gradient of y at a few points and print the values.


import numpy as np

def y(x1, x2, x3):
    return 2 * x1 + 3 * x2 + 3 * x3 + 4


def gradient_y():
    return np.array([2, 3, 3])

points = [
    (0, 0, 0),
    (1, 2, 3),
    (5, 9, 7),
    (2, 5, 6)
]

for p in points:
    x1, x2, x3 = p
    value = y(x1, x2, x3)
    gradient = gradient_y()

    print(f"Point (x1, x2, x3) = {p}")
    print(f"y = {value}")
    print(f"Gradient = {gradient}")
