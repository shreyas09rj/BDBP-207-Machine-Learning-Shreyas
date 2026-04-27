#!/usr/bin/python
# 10 fold cross validation
# Sonar dataset

import pandas as pd
from sklearn.model_selection import  cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline


def data_insert():
    filepath = "sonar data.csv"
    data = pd.read_csv(filepath)
    print(data.head())
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
# feature encoding
# labelencoder for target column
# onehotencoder for features
def encoder(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded


def build_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    return pipeline


def k_fold_cross_val_score(model, X, y):
    # kfold = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    return scores


def main():
    X, y = data_insert()
    y = encoder(y)

    model = build_model()
    scores = k_fold_cross_val_score(model, X, y)

    print("Cross-validation scores:", scores)
    print("Mean Accuracy:", scores.mean())


if __name__ == "__main__":
    main()
