#!/usr/bin/python


from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def data_insert():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test

def model_selection(X_train, y_train, X_val, y_val):

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(max_depth=5)
    }

    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        mse = mean_squared_error(y_val, predictions)

        print(f"{name} Validation MSE: {mse}")

        if mse < best_score:
            best_score = mse
            best_model = model
            best_name = name

    print("\n Best Model:", best_name)
    return best_model

def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Test MSE:", mse)


def main():
    X, y = data_insert()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    best_model = model_selection(X_train, y_train, X_val, y_val)

    test_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
