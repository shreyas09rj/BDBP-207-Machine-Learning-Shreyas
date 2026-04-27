#!/usr/bin/python


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data[:, :2]
y = data.target

mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))