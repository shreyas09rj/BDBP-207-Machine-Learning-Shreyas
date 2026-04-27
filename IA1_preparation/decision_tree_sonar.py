#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def input_data():
    file_path = "sonar data.csv"
    data = pd.read_csv(file_path, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X,y
def label_encoder(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y

def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=15,min_samples_split=4,min_samples_leaf=2,random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict_model(clf, X_test,y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    # classification=classification_report(y_test,y_pred)
    # print("Classification Report")
    # print(classification)




def main():
    X, y = input_data()
    y = label_encoder(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    clf = train_model(X_train, y_train)
    predict_model(clf, X_test,y_test)
if __name__ == '__main__':
    main()













