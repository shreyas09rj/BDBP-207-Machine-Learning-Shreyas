#!/usr/bin/python
# K-fold cross validation
# Implement From scratch ,then use scikit-learn methods.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data():
    file_path = "data.csv"
    data = pd.read_csv(file_path)
    return data


def split_data(data):
    data["diagnosis"] = data["diagnosis"].map({"M":0, "B":1})
    data = data.drop(columns=["id"])
    X=data.drop(columns=["diagnosis"], axis=1)
    y=data["diagnosis"]
    # print(y)
    return X,y

def k_fold(X, y, k=10):

    indices = np.arange(len(X))
    fold_size = len(X) // k
    accuracies = []

    for i in range(k):

        start = i * fold_size
        end = start + fold_size
        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

    print("Accuracy:", np.mean(accuracies))


def main():
    data = load_data()
    print(data.head())
    X ,y = split_data(data)
    k_fold(X, y, k=10)


if __name__ == '__main__':
    main()



