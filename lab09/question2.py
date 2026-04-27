#!/usr/bin/python
# Regression Decision Tree using scikit-learn

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,r2_score


def input_data():
    file_path = "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab09/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    print(data.head())
    return data


def main():
    data = input_data()
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    # print("\nFeature shape:", X.shape)
    # print("Target shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining size:", X_train.shape)
    print("Testing size:", X_test.shape)

    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nMean Squared Error:", mse)
    print("R2:", r2)

if __name__ == "__main__":
    main()