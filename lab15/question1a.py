#!/usr/bin/python
# Gradient Boosting Regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


def load_dataset():
    file_path = "Boston.csv"
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return df, X, y


def eda(df):
    print(df.head())
    print(df.describe())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
    plt.show()


def pre_processing(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def train_split(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=10
    )
    return X_train, X_test, y_train, y_test


def gradient(X_train, y_train, X_test, y_test, X_scaled, y):
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )

    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)

    cv_scores = cross_val_score(gbr, X_scaled, y, cv=10, scoring='r2')
    print("Mean R2:", cv_scores.mean())
    print("Std Dev:", cv_scores.std())


def main():
    df, X, y = load_dataset()
    eda(df)

    X_scaled = pre_processing(X)
    X_train, X_test, y_train, y_test = train_split(X_scaled, y)

    gradient(X_train, y_train, X_test, y_test, X_scaled, y)

if __name__ == "__main__":
    main()