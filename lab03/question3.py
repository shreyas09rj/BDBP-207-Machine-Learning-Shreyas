#!/bin/usr/python3

# Use the above simulated CSV file and implement the following from scratch in Python
# Read simulated data csv file
# Form x and y (disease_score_fluct)
# Write a function to compute hypothesis
# Write a function to compute the cost
# Write a function to compute the derivative
# Write update parameters logic in the main function

import csv
import numpy as np

# 1. Read CSV
def read_csv(filename):
    X = []
    y = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            X.append([float(row[0]), float(row[1]), float(row[2])])
            y.append(float(row[3]))

    return np.array(X), np.array(y)


# Normalize features
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm


# Add bias column
def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def hypothesis(X, theta):
    return np.dot(X, theta)


def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    return (1/(2*m)) * np.sum((predictions - y) ** 2)


def compute_gradient(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    return (1/m) * np.dot(X.T, (predictions - y))


def main():
    X, y = read_csv("/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv")

    # Normalize
    X = normalize(X)

    # Add bias
    X = add_bias(X)

    theta = np.zeros(X.shape[1])
    alpha = 0.01
    iterations = 1000

    for i in range(iterations):
        grad = compute_gradient(X, y, theta)
        theta = theta - alpha * grad

        if i % 100 == 0:
            print("Cost:", compute_cost(X, y, theta))

    print("\nFinal parameters:", theta)


if __name__ == "__main__":
    main()






