#!/usr/bin/python
# Compare result with and without pre-processing

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def load_data():
    file_path = "sonar data.csv"
    data = pd.read_csv(file_path)
    print(data.head())
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
# feature encoding
# labelencoder for target column
# onehotencoder for features
def encoder(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y

def build_model ():
    logistic = LogisticRegression()
    return logistic

def k_cross_val_score(model, X, y):
    kfold = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kfold)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    return scores


def main ():
    X, y = load_data()
    y = encoder(y)
    model = build_model()
    scores = k_cross_val_score(model, X, y)
    print("Cross Validation scores: ",scores)
    print("Accuracy :", scores.mean())
if __name__ == "__main__":
    main()










