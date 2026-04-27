#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_dataset():

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"

    df = pd.read_csv(url, header=None)

    data = df.values

    X = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)

    return X, y, df

def eda(df):
    print(df.head())
    print(df.shape)
    print(df.describe())
    print(df.isnull().sum())

def handle_missing(X):
    X = X.fillna(X.mean())
    return X

def ordinal_encoder(X_train, X_test):
    encoder = OrdinalEncoder()
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)
    return X_train, X_test
def one_hot_encoder(X_train, X_test):
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)
    return X_train, X_test

def train_model (X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    return model
def label_encoder(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    return y_train, y_test

def evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracy)
    calssification_report = classification_report(y_test, y_pred)
    print("classificatuion",calssification_report)
def main():
    X, y, df = load_dataset()
    X = handle_missing(X)
    eda(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train,X_test = (X_train, X_test)
    X_train,X_test  =  one_hot_encoder(X_train, X_test)
    y_train,y_test = label_encoder(y_train, y_test)
    model = train_model(X_train,y_train)
    evaluation(model, X_test, y_test)

if __name__ == "__main__":
    main()















