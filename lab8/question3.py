#!/bin/usr/python


def ordinal_encoding(data, mapping):
    return [mapping[val] for val in data]

data = ["Low", "Medium", "High", "Low"]

mapping = {"Low": 1, "Medium": 2, "High": 3}

encoded = ordinal_encoding(data, mapping)
print(encoded)