#!/usr/bin/python
# load_iris

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report

def data_input():

    X, y = load_iris(return_X_y=True)
    return X, y

def train_model(X_train,X_test, y_train,y_test):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy :",accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))


def main():
    X, y = data_input()
    print(X[:5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_model(X_train,X_test,y_train,y_test)


if __name__ == "__main__":
    main()
















