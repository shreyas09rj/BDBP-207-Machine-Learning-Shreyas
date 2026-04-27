#!/usr/bin/python
# Ada boost (Adapative Boosting)
# Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    iris = load_iris()
    return iris.data, iris.target


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    base = DecisionTreeClassifier(max_depth=1, random_state=42)

    model = AdaBoostClassifier(
        estimator=base,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def evaluation(y_test, prediction):
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("\nClassification Report:\n", classification_report(y_test, prediction))


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    prediction = predict(model, X_test)

    evaluation(y_test, prediction)


if __name__ == '__main__':
    main()













