#!/usr/bin/python
# Information Gain implementation using SONAR dataset

import numpy as np
import pandas as pd
from collections import Counter


def load_data():
    file_path = "sonar data.csv"
    data = pd.read_csv(file_path, header=None)
    print("First 5 rows:")
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


def information_gain(parent_labels, left_labels, right_labels):

    total = len(parent_labels)

    parent_entropy = entropy(parent_labels)

    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)

    weighted_entropy = (
        (len(left_labels) / total) * left_entropy +
        (len(right_labels) / total) * right_entropy
    )

    ig = parent_entropy - weighted_entropy

    return ig


def main():

    data = load_data()

    labels = data.iloc[:, -1]

    feature = data.iloc[:, 0]

    threshold = 0.03

    left = data[feature <= threshold]
    right = data[feature > threshold]

    left_labels = left.iloc[:, -1]
    right_labels = right.iloc[:, -1]

    ig = information_gain(labels, left_labels, right_labels)

    print("\nInformation Gain:", ig)


if __name__ == "__main__":
    main()