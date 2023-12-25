import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy

data = pd.read_csv(r'D:\desktop 1\Users\Nessie\Documents\coding\Machine Learning\Assignment 4\Chemical Composion of Ceramic.csv')
X = data.drop(['Ceramic Name', 'Part'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# K-Means clustering
kval_range = range(1, 10)
inertias = []

for k in kval_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(kval_range, inertias, marker='*', mec = 'c', mfc = 'c', ms = 10)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

optimal = 3

diff = np.diff(inertias)
elbow_index = np.argmin(diff) + 1
# optimal = kval_range[elbow_index]

# kmeans = KMeans(n_clusters=optimal, random_state=42)
labels = kmeans.fit_predict(X_scaled)

colors = plt.cm.viridis(np.linspace(0, 1.5, len(X.columns)))
plt.figure(figsize=(10, 8))

# points = X_pca[labels]
for i, color in zip(range(len(X.columns)), colors):
    # points = X_pca[labels == i]
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], c=color, alpha=0.5, label=f'Chemical {i + 1}')

cluster_colors = ['red', 'black']
for i, color in enumerate(cluster_colors):
    points = X_pca[labels == i]
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    circle = plt.Circle((center[0], center[1]), radius, color=color, fill=False, linestyle='dashed')
    plt.scatter(center[0], center[1], marker='*', s=200, c=color, edgecolor='black', label=f'Centroid {i + 1}')
    plt.gca().add_patch(circle)

plt.xlabel('Chemical composion 1')
plt.ylabel('Chemical composion 2')
plt.title(f'K-Means Clustering for Chemicals in Ceramics in {optimal} Clusters')

plt.legend()
plt.show()

plt.figure(figsize=(15, 8))
dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method = 'ward'))
plt.title('Chemical composition Dendrogram')
plt.xlabel('Chemical composion values')
plt.ylabel('Euclidean Distance')
plt.show()