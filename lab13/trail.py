#!/usr/bin/python
# Random Forest Regression with Hyperparameter Tuning
# Diabetes dataset

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def inputdata():
    data = load_diabetes()
    X = data.data
    y = data.target
    return X, y


def train_model(X_train, X_test, y_train, y_test):

    # Base model
    rf = RandomForestRegressor(random_state=42)

    # Parameter grid for tuning
    param_grid = {
        "n_estimators": [200, 500, 800],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    # Grid Search with cross-validation
    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    predictions = best_model.predict(X_test)

    print("Best Parameters:", grid.best_params_)
    print("Mean Squared Error:", mean_squared_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))


def main():

    X, y = inputdata()

    print("First 5 rows of dataset:")
    print(X[:5])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    train_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()