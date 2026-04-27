#!/usr/bin/python
# Bagging Regressor
# Without scikit-learn

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error



class Bagging:
    def __init__(self,n_estimators=10,max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def bootstrap(self,X,y):
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples,n_samples,replace=True)
        return X[indices],y[indices]
    def fit(self,X,y):
        self.models = []

        for i in range(self.n_estimators):
            X_sample,y_sample = self.bootstrap(X,y)
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X_sample,y_sample)
            self.models.append(model)

    def predict(self,X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions,axis=0)


def input_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X,y


def train_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train,y_train):
    model = BaggingScrath(n_estimators=20,max_depth=5)
    model.fit(X_train,y_train)
    return model

def predict(model,X_test):
    predictions = model.predict(X_test)
    return predictions

def evaluate(y_test,predictions):
    mse = mean_squared_error(y_test,predictions)
    print("MSE:",mse)



def main():
    X,y = input_data()
    X_train, X_test, y_train, y_test = train_split(X,y)
    model = train_model(X_train,y_train)
    predictions = predict(model,X_test)
    evaluate(y_test,predictions)



if __name__ == "__main__":
    main()






