#!/usr/bin/python
# Decision Tree Classification using SONAR dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    file_path = "sonar data.csv"

    data = pd.read_csv(file_path, header=None)

    print("First 5 rows of dataset:")
    print(data.head())

    return data


def main():
    data = load_data()

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()