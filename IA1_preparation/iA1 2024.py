#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




def generate_feature():

    np.random.seed(1)

    # X ~ N(0,1)
    X = np.random.normal(0, 1, 100)

    return X




def generate_noise():

    # e ~ N(0,0.25)
    e = np.random.normal(0, 0.5, 100)

    return e


def generate_target(X, e):

    y = -1 + 0.5 * X + e

    # Length of y = 100
    # theta0 = -1
    # theta1 = 0.5

    return y




def plot_scatter(X, y):

    plt.scatter(X, y)

    plt.title("Scatter Plot of X vs y")

    plt.xlabel("X")

    plt.ylabel("y")

    plt.show()



def train_model(X, y):

    X = X.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=1
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Scatter plot of test data
    plt.scatter(X_test, y_test)

    # Regression line
    plt.plot(X_test, y_pred, color="red")

    plt.title("Linear Regression Fit")

    plt.xlabel("X_test")

    plt.ylabel("y_test")

    plt.show()

    print("\nEstimated Intercept:", model.intercept_)

    print("Estimated Slope:", model.coef_[0])



def main():

    X = generate_feature()

    e = generate_noise()

    y = generate_target(X, e)

    plot_scatter(X, y)

    train_model(X, y)


if __name__ == "__main__":

    main()
