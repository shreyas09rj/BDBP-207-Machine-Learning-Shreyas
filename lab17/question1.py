#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1,13],[1,18],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1],
    [3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]
])

y = np.array(["Blue"]*8 + ["Red"]*8)


def Transform(x):
    x1, x2 = x
    return np.array([
        x1**2,
        np.sqrt(2) * x1 * x2,
        x2**2
    ])

def plot_2d(X, y):
    for i in range(len(X)):
        if y[i] == "Blue":
            plt.scatter(X[i][0], X[i][1], color='blue')
        else:
            plt.scatter(X[i][0], X[i][1], color='red')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original 2D Data")
    plt.show()


def plot_3d(X, y):
    X_t = np.array([Transform(x) for x in X])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X_t)):
        if y[i] == "Blue":
            ax.scatter(*X_t[i], color='blue')
        else:
            ax.scatter(*X_t[i], color='red')

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    plt.title("Transformed 3D Data")
    plt.show()

    return X_t


def high_dim_dot(x1, x2):
    phi_x1 = Transform(x1)
    phi_x2 = Transform(x2)

    return np.dot(phi_x1, phi_x2)


def polynomial_kernel(a, b):
    return (a[0]**2 * b[0]**2 +
            2 * a[0]*b[0]*a[1]*b[1] +
            a[1]**2 * b[1]**2)


def main():


    plot_2d(X, y)
    plot_3d(X, y)

    x1 = np.array([3, 6])
    x2 = np.array([10, 10])

    dot_val = high_dim_dot(x1, x2)
    print("Dot Product in Higher Dimension:", dot_val)

    kernel_val = polynomial_kernel(x1, x2)
    print("Kernel Value:", kernel_val)

    print("Are both equal", dot_val == kernel_val)


if __name__ == "__main__":
    main()