import pandas as pd
import numpy as np

# load data
def load_data():
    file_path = "/home/ibab/PycharmProjects/PythonProject/machinelearning/lab03/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    return data
load_data()
data = load_data()
print(data.head())