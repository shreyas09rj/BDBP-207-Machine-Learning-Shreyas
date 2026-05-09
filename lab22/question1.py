#!/usr/bin/python
# Hierarchical Clustering vs PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram


df = pd.read_csv("NCI60.csv")

df = df.drop(columns=["Unnamed: 0"])

X = df.drop(columns=["labs"])
y = df["labs"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\n--- Dendrogram ---")

X_T = X_scaled.T
subset = X_T[:100]   # take first 100 genes

# Compute linkage matrix
Z = linkage(subset, method='complete')      #Three types are there : Complete , Average , Single

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram (Gene Clustering - subset)")
plt.xlabel("Genes")
plt.ylabel("Distance")
plt.show()


print("\n--- Hierarchical Clustering Feature Reduction ---")

n_gene_clusters = 100

hc = AgglomerativeClustering(
    n_clusters=n_gene_clusters,
    linkage='complete'
)

gene_clusters = hc.fit_predict(X_T)

# Create reduced features
X_hc = np.zeros((X_scaled.shape[0], n_gene_clusters))

for i in range(n_gene_clusters):
    X_hc[:, i] = X_scaled[:, gene_clusters == i].mean(axis=1)

# Train/Test split
X_train_hc, X_test_hc, y_train, y_test = train_test_split(
    X_hc, y, test_size=0.2, random_state=42
)

# Train model
model_hc = RandomForestClassifier(random_state=42,
                                  n_estimators=200,
                                  max_depth=None,
                                  min_samples_split=2,
                                  n_jobs=-1
                                )
model_hc.fit(X_train_hc, y_train)

# Evaluate
y_pred_hc = model_hc.predict(X_test_hc)
acc_hc = accuracy_score(y_test, y_pred_hc)

print("Accuracy (HC):", acc_hc)


print("\n--- PCA Feature Reduction ---")

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Train/Test split
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# Train model
model_pca = RandomForestClassifier(random_state=42,
                                   n_estimators=200,  # number of trees
                                   max_depth=None,  # full depth
                                   min_samples_split=2,
                                   n_jobs=-1  # use all CPU cores
                                   )
model_pca.fit(X_train_pca, y_train)

# Evaluate
y_pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print("Accuracy (PCA):", acc_pca)
