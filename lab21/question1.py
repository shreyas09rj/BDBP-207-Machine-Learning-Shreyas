#!/usr/bin/python


import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k=2, max_iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        labels = []

        
        for point in X:
            distances = [np.linalg.norm(point - c) for c in centroids]
            labels.append(np.argmin(distances))

        labels = np.array(labels)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) for i in range(k)
        ])

    
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


def plot_clusters(X, labels, centroids):
    plt.figure()

    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

    plt.title("K-Means Clustering")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()


X = np.array([
        [1,2],[1.5,1.8],[5,8],
        [8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3]
    ])


centroids, labels = kmeans(X, k=2)

print("Centroids:\n", centroids)
print("Labels:\n", labels)

plot_clusters(X, labels, centroids)






