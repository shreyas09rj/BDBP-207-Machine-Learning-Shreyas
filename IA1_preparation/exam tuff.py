#!/usr/bin/python

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def load_data():

    data = load_iris()

    return data.data, data.target


def decision_tree_model(X, y):

    model = DecisionTreeClassifier()

    scores = cross_val_score(model, X, y, cv=10)

    print("Decision Tree Accuracy:", scores.mean())


def random_forest_model(X, y):

    model = RandomForestClassifier()

    scores = cross_val_score(model, X, y, cv=10)

    print("Random Forest Accuracy:", scores.mean())


def main():

    X, y = load_data()

    decision_tree_model(X, y)

    random_forest_model(X, y)


if __name__ == "__main__":
    main()