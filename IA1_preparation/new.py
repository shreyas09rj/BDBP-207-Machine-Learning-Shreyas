#!/usr/bin/python
# Heart Dataset ML Pipeline

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def load_data():
    file_path = "Heart.csv"
    df = pd.read_csv(file_path)
    return df


def EDA(df):
    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== DATASET INFO =====")
    print(df.info())

    print("\n===== STATISTICS =====")
    print(df.describe())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())


def encode_features(X):
    # One-Hot Encoding for categorical features
    X = pd.get_dummies(X)
    return X


def handle_missing(X):
    # Fill missing values with column mean
    X = X.fillna(X.mean())
    return X


def normalize(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def logistic(X, y):
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    return scores.mean(), scores.std()


def random_forest(X, y):
    model = RandomForestClassifier()
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    return scores.mean(), scores.std()


def compare_models(X, y):

    log_mean, log_std = logistic(X, y)
    rf_mean, rf_std = random_forest(X, y)

    print("\n===== MODEL COMPARISON =====")

    print("\nLogistic Regression")
    print("Mean Accuracy:", log_mean)
    print("Std Deviation:", log_std)

    print("\nRandom Forest")
    print("Mean Accuracy:", rf_mean)
    print("Std Deviation:", rf_std)


def main():

    # Load dataset
    df = load_data()

    # Perform EDA
    EDA(df)

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode categorical features
    X = encode_features(X)

    # Handle missing values
    X = handle_missing(X)

    # Normalize features
    X = normalize(X)

    # Compare models
    compare_models(X, y)


if __name__ == "__main__":
    main()



