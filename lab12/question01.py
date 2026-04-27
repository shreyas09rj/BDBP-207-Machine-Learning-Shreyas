#!/usr/bin/env python3
#Decision Tree Regressor
import numpy as np
from sklearn.datasets import load_diabetes


class TreeNode:
    def __init__(self, feat=None, split=None, left_child=None, right_child=None, output=None):
        self.feat = feat
        self.split = split
        self.left_child = left_child
        self.right_child = right_child
        self.output = output


def calc_var(arr):
    return np.var(arr)


def var_gain(parent, left, right):
    total_len = len(parent)

    w_left = len(left) / total_len
    w_right = len(right) / total_len

    combined_var = w_left * calc_var(left) + w_right * calc_var(right)

    return calc_var(parent) - combined_var


def choose_split(data, target):
    best_score = -1e9
    best_col = None
    best_cut = None

    n_cols = data.shape[1]

    for c in range(n_cols):
        vals = np.unique(data[:, c])

        for t in vals:
            mask_left = data[:, c] <= t
            mask_right = data[:, c] > t

            y_left = target[mask_left]
            y_right = target[mask_right]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            score = var_gain(target, y_left, y_right)

            if score > best_score:
                best_score = score
                best_col = c
                best_cut = t

    return best_col, best_cut


class MyTreeRegressor:
    def __init__(self, depth_limit=5):
        self.depth_limit = depth_limit
        self.tree_root = None

    def train(self, X, y):
        self.tree_root = self._grow_tree(X, y, 0)

    def _grow_tree(self, X, y, level):

        # stopping rules
        if len(y) == 0:
            return TreeNode(output=0)

        if level >= self.depth_limit or len(np.unique(y)) == 1:
            return TreeNode(output=np.mean(y))

        col, cut = choose_split(X, y)

        if col is None:
            return TreeNode(output=np.mean(y))

        left_mask = X[:, col] <= cut
        right_mask = X[:, col] > cut

        left_branch = self._grow_tree(X[left_mask], y[left_mask], level + 1)
        right_branch = self._grow_tree(X[right_mask], y[right_mask], level + 1)

        return TreeNode(feat=col, split=cut,
                        left_child=left_branch,
                        right_child=right_branch)

    def _predict_row(self, row, node):
        if node.output is not None:
            return node.output

        if row[node.feat] <= node.split:
            return self._predict_row(row, node.left_child)
        else:
            return self._predict_row(row, node.right_child)

    def infer(self, X):
        results = []
        for r in X:
            results.append(self._predict_row(r, self.tree_root))
        return np.array(results)


# ---- execution ----

def run():
    dataset = load_diabetes()
    X_data = dataset.data
    y_data = dataset.target

    reg = MyTreeRegressor(depth_limit=3)
    reg.train(X_data, y_data)

    predictions = reg.infer(X_data)

    error = np.mean((predictions - y_data) ** 2)
    print("Mean Squared Error =", error)


if __name__ == "__main__":
    run()
