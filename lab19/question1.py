#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer


data = pd.read_csv("Heart.csv")


y = data["AHD"].astype(str).str.strip().map({"Yes": 1, "No": 0})


X = data.drop(["AHD", "Unnamed: 0"], axis=1, errors='ignore')

X = X.replace("?", np.nan)

X = pd.get_dummies(X, drop_first=True)


imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]


def compute_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # Recall

    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    f1 = (2 * precision * sensitivity) / (precision + sensitivity) \
        if (precision + sensitivity) != 0 else 0

    return accuracy, precision, sensitivity, specificity, f1


thresholds = [0.3, 0.5, 0.7]

print("\nThreshold-wise Metrics:\n")

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    acc, prec, sens, spec, f1 = compute_metrics(y_test.values, y_pred)

    print(f"Threshold = {t}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Sensitivity  : {sens:.4f}")
    print(f"Specificity  : {spec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print("-" * 40)


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

print(f"\nAUC Score: {roc_auc:.4f}")
