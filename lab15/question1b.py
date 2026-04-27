#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def load_input():
    filepath = "Weekly.csv"
    df = pd.read_csv(filepath)
    return df


def eda(df):
    print(df.head())
    print(df.describe())

    df_numeric = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(8,6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()



def preprocessing(df):
    le = LabelEncoder()
    df['Direction'] = le.fit_transform(df['Direction'])
    return df


def split_data(df):
    features = ['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']

    X = df[features]
    y = df['Direction']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    gbc = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )


    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    return gbc



def main():
    df = load_input()

    df = preprocessing(df)

    eda(df)
    X_train, X_test, y_train, y_test = split_data(df)

    train_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()



