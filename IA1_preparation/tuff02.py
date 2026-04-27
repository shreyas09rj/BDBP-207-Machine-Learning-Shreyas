#!/usr/bin/python
# Breast Cancer ML Pipeline

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def load_data():

    data = load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = data.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df




def perform_eda(df):

    print("\nEDA")

    print("\nHead")
    print(df.head())

    print("\nShape")
    print(df.shape)

    print("\nInfo")
    print(df.info())

    print("\nDescribe")
    print(df.describe())

    print("\nMissing values")
    print(df.isnull().sum())



def normalize(X):

    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))

    return X_norm




def gradient_descent(X, y, lr=0.01, epochs=1000):

    m, n = X.shape

    theta = np.zeros(n)

    for i in range(epochs):

        predictions = np.dot(X, theta)

        errors = predictions - y

        gradient = (1/m) * np.dot(X.T, errors)

        theta = theta - lr * gradient

    return theta



def logistic_regression_model(X, y):

    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X, y, cv=10)

    return scores.mean(), scores.std()



def decision_tree_model(X, y):

    model = DecisionTreeClassifier()

    scores = cross_val_score(model, X, y, cv=10)

    return scores.mean(), scores.std()



def random_forest_model(X, y):

    model = RandomForestClassifier()

    scores = cross_val_score(model, X, y, cv=10)

    return scores.mean(), scores.std()


def compare_models(X, y):

    log_mean, log_std = logistic_regression_model(X, y)

    dt_mean, dt_std = decision_tree_model(X, y)

    rf_mean, rf_std = random_forest_model(X, y)

    print("\nModel Comparison\n")

    print("Logistic Regression")
    print("Mean Accuracy:", log_mean)
    print("Std:", log_std)

    print("\nDecision Tree")
    print("Mean Accuracy:", dt_mean)
    print("Std:", dt_std)

    print("\nRandom Forest")
    print("Mean Accuracy:", rf_mean)
    print("Std:", rf_std)




def main():

    df = load_data()

    perform_eda(df)

    X = df.drop("target", axis=1).values
    y = df["target"].values

    # normalize data
    X_norm = normalize(X)

    # run gradient descent
    theta = gradient_descent(X_norm, y)

    print("\nGradient Descent Weights")
    print(theta)

    # train models
    compare_models(X_norm, y)



if __name__ == "__main__":
    main()
