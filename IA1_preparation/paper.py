#!/usr/bin/python

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y


def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def train_model(X, y):
    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X, y, cv=10)

    print("Mean Accuracy:", scores.mean())
    print("Standard Deviation:", scores.std())


def main():

    X, y = load_data()

    X = preprocess_data(X)

    train_model(X, y)


if __name__ == "__main__":
    main()
