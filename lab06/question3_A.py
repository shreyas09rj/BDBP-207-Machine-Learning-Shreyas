#!/usr/bin/python
# Data Standardization

import numpy as np

def data_standardize(X, axis=0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    Z = (X - mean) / std
    return Z

def main():
    X = [[2, 200],
         [3, 300],
         [4, 400],
         [5, 400]]

    V = np.array(X)

    Z = data_standardize(V, axis=0)
    print("Standardized Data:\n", Z)

if __name__ == '__main__':
    main()
