#!/usr/bin/python
#Data Standardization

import pandas as pd


def load_data():
    file_path = "/exam preparation/data.csv"
    df = pd.read_csv(file_path)
    return df



def split_data(df):
    df["diagnosis"] = df["diagnosis"].map({"M":0, "B":1})
    df = df.drop(columns=["id"])
    X=df.drop(columns=["diagnosis"], axis=1)
    y=df["diagnosis"]
    # print(y)
    return X,y

def data_standardize(X ):
    mean = X.mean()
    std = X.std()
    z = (X - mean) / std
    return z



def main():
    df = load_data()
    X , y = split_data(df)
    standardized_data = data_standardize(X)
    print(f"Standardize data:\n",standardized_data.mean(),"\n")
    print(f"Standardize data:\n",standardized_data.std(),"\n")
    print(f"Standardize data:\n",standardized_data)

if __name__ == '__main__':
    main()