import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load data
data = pd.read_csv(r'D:\desktop 1\Users\Nessie\Documents\coding\Machine Learning\Assignment 4\Chemical Composion of Ceramic.csv')
X = data.drop(['Ceramic Name', 'Part'], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
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
plt.plot(kval_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

diff = np.diff(inertias)
elbow_index = np.argmin(diff) + 1
optimal_k = kval_range[elbow_index]

# # K-Means clustering
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

colors = plt.cm.viridis(np.linspace(0, 1.5, len(X.columns)))
plt.figure(figsize=(10, 8))

# cluster_points = X_pca[labels]
for i, color in zip(range(len(X.columns)), colors):
    # cluster_points = X_pca[labels == i]
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], c=color, alpha=0.5, label=f'Chemical {i + 1}')

# Plot cluster centers with distinct colors (red and black)
cluster_colors = ['red', 'black']
for i, color in enumerate(cluster_colors):
    cluster_points = X_pca[labels == i]
    cluster_center = np.mean(cluster_points, axis=0)

    cluster_radius = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))

    circle = plt.Circle((cluster_center[0], cluster_center[1]), cluster_radius, color=color, fill=False, linestyle='dashed')
    plt.scatter(cluster_center[0], cluster_center[1], marker='*', s=200, c=color, edgecolor='black', label=f'Centroid {i + 1}')
    plt.gca().add_patch(circle)

plt.xlabel('chemical')
plt.ylabel('chemical')
plt.title(f'K-Means Clustering for Chemicals in Ceramics in {optimal_k} Clusters')

plt.legend()
plt.show()
