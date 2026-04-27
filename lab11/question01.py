#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_iris
from collections import Counter


def calc_entropy(labels):
    freq = np.bincount(labels)
    probs = freq / len(labels)
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def gain(parent_y, left_y, right_y):
    total = len(parent_y)
    w_left = len(left_y) / total
    w_right = len(right_y) / total

    return calc_entropy(parent_y) - (
        w_left * calc_entropy(left_y) + w_right * calc_entropy(right_y)
    )


def find_split(features, labels):
    best_score = -1
    best_feature = None
    best_threshold = None

    n_features = features.shape[1]

    for f in range(n_features):
        unique_vals = np.unique(features[:, f])

        for threshold in unique_vals:
            left_mask = features[:, f] <= threshold
            right_mask = features[:, f] > threshold

            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            score = gain(labels, left_labels, right_labels)

            if score > best_score:
                best_score = score
                best_feature = f
                best_threshold = threshold

    return best_feature, best_threshold


def most_common_label(labels):
    return Counter(labels).most_common(1)[0][0]


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root_node = None

    def _grow(self, X, y, depth):
        # stopping conditions
        if len(set(y)) == 1 or depth >= self.max_depth:
            return TreeNode(label=most_common_label(y))

        feat, thresh = find_split(X, y)

        if feat is None:
            return TreeNode(label=most_common_label(y))

        left_mask = X[:, feat] <= thresh
        right_mask = X[:, feat] > thresh

        left_branch = self._grow(X[left_mask], y[left_mask], depth + 1)
        right_branch = self._grow(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature=feat, threshold=thresh,
                        left=left_branch, right=right_branch)

    def fit(self, X, y):
        self.root_node = self._grow(X, y, 0)

    def _predict_single(self, sample, node):
        if node.label is not None:
            return node.label

        if sample[node.feature] <= node.threshold:
            return self._predict_single(sample, node.left)
        else:
            return self._predict_single(sample, node.right)

    def predict(self, X):
        results = []
        for sample in X:
            results.append(self._predict_single(sample, self.root_node))
        return np.array(results)


# ---- run model ----

iris_data = load_iris()
features = iris_data.data
targets = iris_data.target

clf = DecisionTree(max_depth=3)
clf.fit(features, targets)

predictions = clf.predict(features)

accuracy = np.mean(predictions == targets)
print("Accuracy =", accuracy)
