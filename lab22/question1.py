#!/usr/bin/python
# Hierarchical Clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

nci60 = load_data('NCI60')
nci_labs = nci60['labels']
nci_data = nci60['data']

print("Shape of data:", nci_data.shape)
print(nci_labs.value_counts())

scaler = StandardScaler()
nci_scaled = scaler.fit_transform(nci_data)

pca = PCA()
nci_scores = pca.fit_transform(nci_scaled)




