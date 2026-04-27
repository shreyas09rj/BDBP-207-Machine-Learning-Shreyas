#!/usr/bin/python
# Wincson Breast Cancer

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report


def load_datset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X,y
def eda(df):
    print("First 5 rows")
    print(df.head())

    print("\n shape :",df.shape)
    print("\n describe :",df.describe())
    print("\n missing values",df.isnull().sum())


def preprocessing(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
def ridge_classifier(X_train, y_train):
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
    model.fit(X_train,y_train)
    return model

def lasso_classifier(X_train, y_train):
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    model.fit(X_train,y_train)
    return model
def evaluation(model,X_test,y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy:",accuracy)
    classification = classification_report(y_test,y_pred)
    print("Classification Report:",classification_report)

def main():
    X,y = load_datset()
    df = pd.concat([X,y],axis=1)
    eda(df)
    X_scaled = preprocessing(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
    print("Ridge Regression :")
    ridge_model = ridge_classifier(X_train, y_train)
    print("Ridge Regression Accuracy:",ridge_model.score(X_train,y_train))
    lasso_model = lasso_classifier(X_train, y_train)
    print("Lasso Regression Accuracy:",lasso_model.score(X_train,y_train))

if __name__ == "__main__":
    main()










