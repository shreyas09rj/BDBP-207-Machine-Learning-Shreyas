#!/usr/bin/python
#california
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
def load_housing():
    housing_data = fetch_california_housing(as_frame=True)
    X = housing_data.data
    y = housing_data.target

    # X = pd.DataFrame(X, columns=housing_data.feature_names)
    # y = pd.DataFrame(y, columns=["target"])

    return X, y


def encode_features(X):
    X = X.fillna(X.mean())
    return X
def eda(df):
    print(df.describe())
    print(df.head())
    print(df.shape)
    print(df.index)
    print(df.columns)
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.info)
def preprocessing(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
def linear_regression(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model
from sklearn.ensemble import RandomForestRegressor

# def random_forest(X_train, y_train):
#
#     model = RandomForestRegressor(n_estimators=200, random_state=10)
#
#     model.fit(X_train, y_train)
#
#     return model
def evaluation(model, X_test, y_test):
    y_pred=model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    print("MSE:",mse)
    print("R2:",r2)


def main():
    X, y  = load_housing()
    df = pd.concat([X,y],axis=1)
    X=encode_features(X)
    X_scaled = preprocessing(X)
    eda(df)
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=42)
    model = linear_regression(X_train,y_train)
    evaluation(model,X_test,y_test)

if __name__ == "__main__":
    main()






















