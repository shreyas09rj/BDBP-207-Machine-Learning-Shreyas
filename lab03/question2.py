#!/usr/bin/python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def load_data():
    df = pd.read_csv(
        "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop("disease_score_fluct", axis=1)
    y = df["disease_score_fluct"]
    return X, y


def train_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def model_fit(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def evaluation(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def main():
    # Step 1: Load data
    X, y = load_data()

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_split(X, y)

    # Step 3: Train model
    model = model_fit(X_train, y_train)

    # Step 4: Predict
    y_pred = predict(model, X_test)

    # Step 5: Evaluate
    mse, r2 = evaluation(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)

if __name__ == "__main__":
    main()
