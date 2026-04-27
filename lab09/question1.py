#!/usr/bin/python
# Partition a datset
# Different threshold value
import numpy as np
import pandas as pd

def input_data():
    file_path= "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab09/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    print(data.head())
    return data

def partition(data , threshold):
    left = data[data ["BP"] <= threshold]
    right = data[data ["BP"] > threshold]
    return left , right

def compute_mse(partition):
    if len(partition)==0:
        return 0
    y = partition.iloc[:,-1]
    mean = np.mean(y)
    mse = np.mean((y-mean)**2)
    return mse

def evaluation(threshold, left,right):
    print(f"threshold :", threshold)
    print(f"left :", len(left))
    print(f"right :", len(right))

    left_mse = compute_mse(left)
    right_mse = compute_mse(right)

    print(f"left mse: {left_mse}")
    print(f"right mse: {right_mse}")

    weighted_mse = (
    (len(left) * left_mse + len(right) * right_mse) / (len(left) + len(right))
    )

    print(f"weighted mse: {weighted_mse}")



def main():
    data = input_data()
    print("Data shape:", data.shape)

    thresholds = [80, 72, 82]

    for t in thresholds:
        left,right = partition(data , t)
        evaluation(t,left,right)
    # left , right = partition(data , 78)
    # left , right = partition(data , 82)

    # print(f"left ",left)
    # print(f"right ",right)





if __name__ == "__main__":
    main()
