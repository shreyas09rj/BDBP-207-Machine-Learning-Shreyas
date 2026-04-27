#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


X = np.array([
    [6,5],[6,9],[8,6],[8,8],[8,10],[9,2],[9,5],[10,10],
    [10,13],[11,5],[11,8],[12,6],[12,11],[13,4],[14,8]
])

labels = ["Blue","Blue","Red","Red","Red","Blue","Red","Red",
          "Blue","Red","Red","Red","Blue","Blue","Blue"]


y = np.array([0 if l=="Blue" else 1 for l in labels])

def plot_boundary(model, X, y, title):
    h = 0.2
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    for i in range(len(X)):
        color = 'blue' if y[i]==0 else 'red'
        plt.scatter(X[i,0], X[i,1], color=color)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


rbf_model = SVC(kernel='rbf', gamma=0.5, C=1)
rbf_model.fit(X, y)

plot_boundary(rbf_model, X, y, "RBF Kernel")


poly_model = SVC(kernel='poly', degree=2, C=1)
poly_model.fit(X, y)

plot_boundary(poly_model, X, y, "Polynomial Kernel")