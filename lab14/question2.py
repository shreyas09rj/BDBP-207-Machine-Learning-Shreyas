#!/usr/bin/python

import numpy as np
from sklearn.datasets import load_iris


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature] < self.threshold] = -1
        else:
            predictions[X[:, self.feature] < self.threshold] = 1

        return predictions



class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        n_samples, n_features = X.shape

       
        w = np.full(n_samples, (1 / n_samples))

        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

          
            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        preds = np.ones(n_samples)

                        if polarity == 1:
                            preds[X[:, feature] < threshold] = -1
                        else:
                            preds[X[:, feature] < threshold] = 1
                        error = np.sum(w[y != preds])
                        if error > 0.5:
                            error = 1 - error
                            polarity *= -1
                        if error < min_error:
                            stump.feature = feature
                            stump.threshold = threshold
                            stump.polarity = polarity
                            min_error = error


            eps = 1e-10
            alpha = 0.5 * np.log((1 - min_error + eps) / (min_error + eps))


            preds = stump.predict(X)
            w *= np.exp(-alpha * y * preds)
            w /= np.sum(w)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)

        return np.sign(final_pred)


def main():
    data = load_iris()
    X, y = data.data, data.target

    y = np.where(y == 0, -1, 1)

    model = AdaBoost(n_estimators=10)
    model.fit(X, y)

    preds = model.predict(X)

    accuracy = np.mean(preds == y)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
