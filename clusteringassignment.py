# the task I am trying to do is see which chemicals react similarly with
# similar values in order to make ceramics
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'D:\desktop 1\Users\Nessie\Documents\coding\Machine Learning\Assignment 4\Chemical Composion of Ceramic.csv')
X = data.drop(['Ceramic Name', 'Part'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# K-Means clustering
kval = range(1, 10)
inertias = []

for k in kval:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

labels = kmeans.fit_predict(X_scaled)
colors = plt.cm.viridis(np.linspace(0, 1, len(X.columns)))
plt.figure(figsize=(10, 8))
for i, color in zip(range(len(X.columns)), colors):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], c=color, alpha=0.5, label=f'Chemical {i + 1}')

plt.xlabel('chemical')
plt.ylabel('chemical')
plt.title('K-Means Clustering for Chemicals in Ceramics (PCA Space)')

plt.xlim(X_pca[:, 0].min(), X_pca[:, 0].max())
plt.ylim(X_pca[:, 1].min(), X_pca[:, 1].max())

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