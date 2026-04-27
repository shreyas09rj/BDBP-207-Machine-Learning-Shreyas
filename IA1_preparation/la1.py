#!/usr/bin/python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data():

    file_path = "Heart.csv"

    df = pd.read_csv(file_path)

    # remove unnecessary column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    print("\nColumns in dataset:")
    print(df.columns)

    return df



def perform_eda(df):

    print("\nFirst 5 rows")
    print(df.head())

    print("\nDataset Info")
    print(df.info())

    print("\nSummary Statistics")
    print(df.describe())

    print("\nMissing Values")
    print(df.isnull().sum())

    df['AHD'].value_counts().plot(kind='bar')
    plt.title("Target Distribution")
    plt.xlabel("Heart Disease")
    plt.ylabel("Count")
    plt.show()
    # Target distribution
    # plt.figure()
    # sns.countplot(x="AHD", data=df)
    # plt.title("Target Distribution")
    # plt.show()
    #
    # # Correlation heatmap
    # plt.figure(figsize=(10,8))
    # sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    # plt.title("Correlation Matrix")
    # plt.show()
    plt.boxplot(df['Chol'])
    plt.title("Cholesterol Distribution")
    plt.show()


def preprocess_data(df):

    # convert target to numeric
    df["AHD"] = df["AHD"].map({"No":0, "Yes":1})

    X = df.drop("AHD", axis=1)
    y = df["AHD"]

    # encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    X=X.fillna(X.mean())

    # feature scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y



def logistic_model(X, y):

    model = LogisticRegression(max_iter=10)

    scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")

    print("\nLogistic Regression Results")
    print("Mean Accuracy:", scores.mean())
    print("Std Dev:", scores.std())



def random_forest_model(X, y):

    model = RandomForestClassifier(n_estimators=10, random_state=42)

    scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")

    print("\nRandom Forest Results")
    print("Mean Accuracy:", scores.mean())
    print("Std Dev:", scores.std())


def decision_tree_model(X, y):

    model = DecisionTreeClassifier()

    scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")

    print("\nDecision Tree Results")
    print("Mean Accuracy:", scores.mean())
    print("Std Dev:", scores.std())



def main():

    df = load_data()

    perform_eda(df)

    X, y = preprocess_data(df)

    logistic_model(X, y)

    random_forest_model(X, y)

    decision_tree_model(X, y)


if __name__ == "__main__":
    main()








































