#!/usr/bin/python
# heart.csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def input_data():
    file_path = "Heart.csv"
    df = pd.read_csv(file_path)
    return df
def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def missing_values(X):
    X= pd.get_dummies(X)
    return X
def feature_encode(X):
    X = X.fillna(X.mean())
    return X


def label_encode(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y
def eda(df):
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.isnull().sum())
    print(df.info())


def preprocess(X):
    scalar = StandardScaler()
    X_scalar = scalar.fit_transform(X)
    return X_scalar

def logistic(X_scalar, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scalar, y)
    return model

def decision_tree(X_scalar, y):
    model = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)
    model.fit(X_scalar, y)
    return model
def random_forest(X_scalar, y):
    model = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)
    model.fit(X_scalar, y)
    return model
def compare_models(X_scalar,y):
    print("logistic")
    log_model = logistic(X_scalar,y)
    scores = cross_val_score(log_model, X_scalar, y, cv=10, scoring='accuracy')
    print("Accuracy:", scores.mean())

    print("decision tree")
    decision= decision_tree(X_scalar,y)
    scores = cross_val_score(decision, X_scalar, y, cv=10, scoring='accuracy')
    print("Accuracy:", scores.mean())

    print("random forest")
    random = random_forest(X_scalar,y)
    scores= cross_val_score(random, X_scalar, y, cv=10, scoring='accuracy')
    print("Accuracy:", scores.mean())




def main():
    df = input_data()
    X, y = split_data(df)
    X = missing_values(X)
    X= feature_encode(X)
    y = label_encode(y)
    eda(df)
    X_scalar = preprocess(X)
    compare_models(X_scalar,y)


if __name__ == "__main__":
    main()






















