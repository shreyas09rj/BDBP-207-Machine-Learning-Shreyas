#!/usr/bin/python
# Breast Cancer

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
    df['target'] = y

    return df


def perform_eda(df):

    print("Performing EDA")

    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.isnull().sum())


def normalization(X):

    print("Normalizing")

    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))

    return X_norm

def logistic_regression(X, y):

    print("Logistic Regression")

    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

    return scores.mean(), scores.std()


def decision_tree(X, y):

    print("Decision Tree")

    model = DecisionTreeClassifier()

    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

    return scores.mean(), scores.std()


def random_forest(X, y):

    print("Random Forest")

    model = RandomForestClassifier()

    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

    return scores.mean(), scores.std()



def compare_models(X_norm, y):

    print("Comparing Models")

    log_mean, log_std = logistic_regression(X_norm, y)
    decision_tree_mean, decision_tree_std = decision_tree(X_norm, y)
    random_forest_mean, random_forest_std = random_forest(X_norm, y)

    print("\nLogistic Regression")
    print(log_mean, log_std)

    print("\nDecision Tree")
    print(decision_tree_mean, decision_tree_std)

    print("\nRandom Forest")
    print(random_forest_mean, random_forest_std)


def main():

    df = load_data()

    perform_eda(df)

    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_norm = normalization(X)

    compare_models(X_norm, y)


if __name__ == "__main__":
    main()
