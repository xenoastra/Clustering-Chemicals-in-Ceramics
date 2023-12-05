# the task I am trying to do is see which chemicals react similarly with
# similar values in order to make ceramics
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'D:\desktop 1\Users\Nessie\Documents\coding\Machine Learning\Assignment 4\Chemical Composion of Ceramic.csv')

# Extract features for PCA
X = data.drop(['Ceramic Name', 'Part', 'P2O5', 'MnO', 'Rb2O', 'SrO', 'ZnO', 'ZrO2', 'PbO2', 'CuO', 'Y2O3', 'SiO2', 'Al2O3', 'CaO'], axis=1)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Visualize in PCA space
colors = plt.cm.viridis(np.linspace(0, 0.8, len(X.columns)))
plt.figure(figsize=(10, 8))
for i, color in zip(range(len(X.columns)), colors):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], c=color, alpha=0.5, label=f'Chemical {i + 1}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering for Chemicals in Ceramics (PCA Space)')
plt.legend()
plt.show()
# #Hierarcial clustering

# Y = hierarchy.linkage(X.T, 'ward')  # Transpose X to have chemicals as rows and features as columns
# plt.figure(figsize=(10, 5))

# # Define a function to assign colors based on k-means labels
# def get_link_color(index):
#     return plt.cm.viridis(labels[index] / (len(np.unique(labels)) - 1))

# dendrogram = hierarchy.dendrogram(Y, labels=X.columns, link_color_func=get_link_color)

# plt.title('Hierarchical Clustering Dendrogram for Chemical Composition of Ceramics')
# plt.xlabel('Chemicals')
# plt.show()