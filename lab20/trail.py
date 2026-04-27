import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ISLP import load_data


dataset = load_data('NCI60')

X = dataset['data']      
y = dataset['labels']    

X_scaled = StandardScaler().fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', edgecolors='k')

plt.title('PCA  K-Means ')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()
