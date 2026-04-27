#!/usr/bin/python

import numpy as np
from sklearn.datasets import load_iris
from collections import Counter


def entropy(y):
    count = np.bincount(y)
    prob = count / len(y)
    return -np.sum([p * np.log2(p) for p in prob if p > 0])


def info_gain(parent, left, right):
    n = len(parent)
    return entropy(parent) - (len(left)/n)*entropy(left) - (len(right)/n)*entropy(right)


def best_split(X, y):
    best_gain = 0
    split_col = None
    split_val = None

    for col in range(X.shape[1]):
        values = np.unique(X[:, col])

        for val in values:
            left = y[X[:, col] <= val]
            right = y[X[:, col] > val]

            if len(left) == 0 or len(right) == 0:
                continue

            gain = info_gain(y, left, right)

            if gain > best_gain:
                best_gain = gain
                split_col = col
                split_val = val

    return split_col, split_val


def majority(y):
    return Counter(y).most_common(1)[0][0]


class Node:
    def __init__(self, col=None, val=None, left=None, right=None, result=None):
        self.col = col
        self.val = val
        self.left = left
        self.right = right
        self.result = result


class Tree:
    def __init__(self, depth=3):
        self.depth = depth

    def build(self, X, y, d):
        if len(set(y)) == 1 or d == self.depth:
            return Node(result=majority(y))

        col, val = best_split(X, y)

        if col is None:
            return Node(result=majority(y))

        left_idx = X[:, col] <= val
        right_idx = X[:, col] > val

        left = self.build(X[left_idx], y[left_idx], d+1)
        right = self.build(X[right_idx], y[right_idx], d+1)

        return Node(col, val, left, right)

    def fit(self, X, y):
        self.root = self.build(X, y, 0)

    def predict_one(self, x, node):
        if node.result is not None:
            return node.result

        if x[node.col] <= node.val:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])



data = load_iris()
X = data.data
y = data.target

model = Tree(depth=3)
model.fit(X, y)

pred = model.predict(X)

acc = np.mean(pred == y)
print("Accuracy:", acc)
