#!/usr/bin/python
# Implement gradient descent algorithm from scratch using Python

import pandas as pd
import numpy as np


def datainput():
    file_path = "data.csv"
    df = pd.read_csv(file_path)
    return df


def split_data(df):

    df = df.drop(columns=["id", "diagnosis"])

    X = df.drop("fractal_dimension_worst", axis=1)
    y = df["fractal_dimension_worst"]

    X = X.values
    y = y.values

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, y

def eda(df):
    print("Performing EDA")
    print(df.head())
    print("Shape:", df.shape)
    print(df.describe())
    print("Missing values:")
    print(df.isnull().sum())


def gradient(X, y, learning_rate=0.01, epochs=10000):

    print("Performing Gradient Descent")

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0

    for i in range(epochs):

        y_pred = np.dot(X, weights) + bias

        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        loss = np.mean((y_pred - y) ** 2)

        if i % 1000 == 0:
            print("Epoch:", i, "Loss:", loss)

    return weights, bias


def predict(X, weights, bias):
    return np.dot(X, weights) + bias


def main():

    df = datainput()

    eda(df)

    X, y = split_data(df)

    weights, bias = gradient(X, y, learning_rate=0.01, epochs=10000)

    y_pred = predict(X, weights, bias)

    print("Predicted values:")
    print(y_pred)


if __name__ == "__main__":
    main()