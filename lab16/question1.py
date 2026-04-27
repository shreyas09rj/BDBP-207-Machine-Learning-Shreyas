#!/usr/bin/python

import numpy as np
from sklearn.tree import DecisionTreeRegressor



X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([2,4,6,8,10,12])



n_trees = 5
trees = []

for i in range(n_trees):
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, y)
    trees.append(tree)


def aggregate_predictions(X, trees):
    preds = []

    for tree in trees:
        preds.append(tree.predict(X))

    preds = np.array(preds)

    final_pred = np.mean(preds, axis=0)

    return final_pred


X_test = np.array([[7],[8]])

final_prediction = aggregate_predictions(X_test, trees)

print("Final Prediction:", final_prediction)
