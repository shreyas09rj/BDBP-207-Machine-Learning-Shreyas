#!/usr/bin/python
# Entropy calculation for a dataset

import numpy as np
import pandas as pd
from collections import Counter


def load_data():
    file_path = "sonar data.csv"
    data = pd.read_csv(file_path, header=None)
    print("First 5 rows of dataset:")
    print(data.head())

    return data


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    entropy_value = 0

    for count in counts.values():
        probability = count / total
        entropy_value -= probability * np.log2(probability)

    return entropy_value


def main():

    data = load_data()
    labels = data.iloc[:, -1]
    ent = entropy(labels)
    print("Entropy of dataset:", ent)


if __name__ == "__main__":
    main()