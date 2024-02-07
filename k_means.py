import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data for clustering
np.random.seed(42)
X = np.random.rand(100, 2)

# Choose the number of clusters
k = 3

# Train a K-Means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get cluster assignments for each data point
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red')
plt.title(f'K-Means Clustering with {k} clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
