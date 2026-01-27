# !/usr/bin/python3


# Implement a linear regression model using scikit-learn for the simulated dataset
#simulated_data_multiple_linear_regression_for_ML.csv  - to predict the “disease_score” from multiple clinical parameters.


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# y = beta0 + beta1* x1 + beta2 * x2 + ...
#load data set
def load_data():
    file_path = "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    return data

# split features and target
def split_features(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# split into training and testing data
def split_train_test(X, y, test_size=0.2, random_state=99):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train linear regression model
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred


def main():
    data = load_data()
    print(data.head())
    target_column = "disease_score"
    X, y = split_features(data, target_column)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train_linear_regression(X_train, y_train)

    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print("MSE:", mse)
    print("R2:", r2)



if __name__=="__main__":
    main()
