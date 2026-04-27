#!/usr/bin/python
# Diabetes Regression using Random Forest

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data():

    data = load_diabetes(as_frame=True)

    X = data.data
    y = data.target

    return X, y

def eda(df):

    print("First 5 rows")
    print(df.head())

    print("\nShape:", df.shape)

    print("\nSummary statistics")
    print(df.describe())

    print("\nMissing values")
    print(df.isnull().sum())

    print("\nDataset Info")
    df.info()



def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model



def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print("\nMean Squared Error:", mse)

    print("R2 Score:", r2)



def main():

    X, y = load_data()

    df = pd.concat([X, y], axis=1)

    eda(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
