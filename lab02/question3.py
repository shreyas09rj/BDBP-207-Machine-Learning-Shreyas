



import random
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def load_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X, y


def divide_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def initialize_model():
    return LinearRegression()


def main():
    # 1. Load data
    X, y = load_data()

    # 2. Divide data
    X_train, X_test, y_train, y_test = divide_data(X, y)

    # 3. Standardize
    X_train, X_test = standardize_data(X_train, X_test)

    # 4. Initialize model
    model = initialize_model()

    # 5. Train model
    model.fit(X_train, y_train)

    # 6. Test model
    y_pred = model.predict(X_test)

    # 7. Print R2
    print("R2 Score:", r2_score(y_test, y_pred))


if __name__ == "__main__":
    main()
