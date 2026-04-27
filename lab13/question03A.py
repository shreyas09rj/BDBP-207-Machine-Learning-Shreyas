#!/usr/bin/python
# Random Forest Regression using GridSearchCV
# Diabetes Dataset

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def input_data():
    data = load_diabetes()
    X = data.data
    y = data.target
    return X, y


def train_model(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nMean Squared Error:", mean_squared_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))


def main():

    X, y = input_data()

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




