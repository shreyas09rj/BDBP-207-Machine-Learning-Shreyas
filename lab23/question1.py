#!/usr/bin/python
# Generative models
# iris.cv dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier



def load_data():
    df = pd.read_csv("Iris.csv")
    X = df[['SepalLengthCm','SepalWidthCm']]
    y = df[['Species']]
    return X, y

def noise(X):
    np.random.seed(42)

    noise1=np.random.normal(0,0.3,size=len(X))
    noise2 = np.random.normal(0,0.3,size=len(X))
    X_noise = X.copy()
    X_noise['SepalLengthCm'] += noise1
    X_noise['SepalWidthCm'] += noise2
    return X_noise

def discritize(X):


def main():
    X, y = load_data()
    X_noise = noise(X)

if __name__ == "__main__":
    main()