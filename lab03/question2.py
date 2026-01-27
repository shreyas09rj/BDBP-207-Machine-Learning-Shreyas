#!/usr/bin/python3

# Implement Multiple Linear Regression using scikit-learn
# Predict disease_score_fluct from multiple clinical parameters

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Load dataset
def load_data():
    file_path = "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    return data


# Split features and target
def split_features(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


# Split into training and testing data
def split_train_test(X, y, test_size=0.3, random_state=99):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Train linear regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred


def main():
    # Load data
    data = load_data()
    print("Dataset Preview:")
    print(data.head(), "\n")

    # Define target
    target_column = "disease_score_fluct"

    # Feature-target split
    X, y = split_features(data, target_column)

    # Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Train model
    model = train_linear_regression(X_train, y_train)

    # Evaluate model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)

    print("Model Performance:")
    print(f"MSE: {mse:.3f}")
    print(f"R2 Score: {r2:.3f}")

if __name__ == "__main__":
    main()

