#!/usr/bin/python

import csv

def load_data(filename):
    X = []
    y = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        target_index = header.index("disease_score_fluct")

        for row in reader:
            row = [float(val) for val in row]
            y.append(row[target_index])
            X.append(row[:target_index] + row[target_index+1:])

    return X, y, header


def normalize_features(X):
    X_norm = [[0]*len(X[0]) for _ in range(len(X))]

    for j in range(len(X[0])):
        col = [row[j] for row in X]
        mean = sum(col) / len(col)
        std = (sum((v - mean) ** 2 for v in col) / len(col)) ** 0.5

        for i in range(len(X)):
            X_norm[i][j] = (X[i][j] - mean) / std

    return X_norm


def hypothesis_theta(x, theta):
    result = theta[0]
    for i in range(len(x)):
        result += theta[i+1] * x[i]
    return result


def cost_function(X, y, theta):
    m = len(X)
    return sum((hypothesis_theta(X[i], theta) - y[i])**2 for i in range(m)) / m


def compute_gradients(X, y, theta):
    m = len(X)
    gradients = [0] * len(theta)

    for i in range(m):
        error = hypothesis_theta(X[i], theta) - y[i]
        gradients[0] += error
        for j in range(len(X[i])):
            gradients[j+1] += error * X[i][j]

    return [g/m for g in gradients]


def r2_score_scratch(X, y, theta):
    preds = [hypothesis_theta(X[i], theta) for i in range(len(X))]
    y_mean = sum(y)/len(y)

    ss_res = sum((y[i]-preds[i])**2 for i in range(len(y)))
    ss_tot = sum((y[i]-y_mean)**2 for i in range(len(y)))

    return 1 - ss_res/ss_tot


def main():
    filename = "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv"

    X, y, header = load_data(filename)
    X = normalize_features(X)

    theta = [0]*(len(X[0])+1)
    alpha = 0.01
    epochs = 1000

    for epoch in range(epochs):
        gradients = compute_gradients(X, y, theta)
        for j in range(len(theta)):
            theta[j] -= alpha * gradients[j]

        if epoch % 100 == 0:
            print("Epoch", epoch, "Cost:", cost_function(X, y, theta))

    print("\nFinal R2:", r2_score_scratch(X, y, theta))


if __name__ == "__main__":
    main()
