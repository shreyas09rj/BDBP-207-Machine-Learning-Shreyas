#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [3,4],[2,2],[4,4],[1,4],   # Red
    [2,1],[4,3],[4,1]          # Blue
])

y = np.array([1,1,1,1, -1,-1,-1])


def plot_data():
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i,0], X[i,1], color='red', label='Red' if i==0 else "")
        else:
            plt.scatter(X[i,0], X[i,1], color='blue', label='Blue' if i==4 else "")

plt.figure()
plot_data()
plt.title("(a) Original Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

plt.figure()
plot_data()

# Optimal hyperplane: x2 = 1.5
x_vals = np.linspace(0,5,100)
plt.plot(x_vals, [1.5]*len(x_vals), 'k-', label="Optimal Hyperplane")


plt.plot(x_vals, [2]*len(x_vals), 'k--', label="Margin")
plt.plot(x_vals, [1]*len(x_vals), 'k--')


support_vectors = np.array([[2,2],[2,1]])
plt.scatter(support_vectors[:,0], support_vectors[:,1],
            s=200, facecolors='none', edgecolors='black', label="Support Vectors")

plt.title("(b)(d)(e) Hyperplane + Margins + Support Vectors")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

plt.figure()
plot_data()

plt.plot(x_vals, [2]*len(x_vals), 'g-', label="Non-optimal Hyperplane")

plt.title("(g) Non-optimal Hyperplane")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()


plt.figure()
plot_data()

plt.scatter(3,2, color='blue', s=100, marker='x', label="New Blue Point")

plt.title("(h) Non-separable Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()


print("Classification Rule:")
print("Red if X2 > 1.5")
print("Blue if X2 < 1.5")

print("\nHyperplane parameters:")
print("beta0 = -1.5, beta1 = 0, beta2 = 1")
