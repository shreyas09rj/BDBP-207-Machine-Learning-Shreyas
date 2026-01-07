#!/usr/bin/python




# Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
# What is the value of x1 at which the function value (y) is zero. What do you infer from this?



import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(-10, 10, 100)
y = x1**2

plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y = x1^2")
plt.grid(True)
plt.show()

points = np.array([-5, -3, 0, 3, 5])

derivative = 2 * points

for i in range(len(points)):
    print(f"x1 = {points[i]}, dy/dx1 = {derivative[i]}")




