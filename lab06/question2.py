#!/usr/bin/python
#Data Normalization
import numpy as np

def data_normalize(X):
    x_min = np.min(X,axis=0)
    x_max = np.max(X, axis=0)

    denominator = x_max - x_min
    denominator[denominator==0]=1
    x_norm = (X - x_min) / denominator
    return x_norm


def main():
    X = [[2 , 200],
         [3, 300],
         [4, 400],
         [5, 400]
         ]
    M = data_normalize(X)
    print(f"Orginal\n",np.array(X))
    print(f"Normalized\n",M)

if __name__ == '__main__':
    main()