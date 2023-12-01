# the task I am trying to do is see which chemicals react similarly with
# similar values in order to make ceramics

import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv(r'c:\Users\Vanessa\Documents\Visual Studio Code\Machine Learning\Chemical Composion of Ceramic.csv')
# print (data.head(100))
X = data.drop(['Ceramic Name', 'Part'], axis=1)
# print (X.head(100))

#Hierarcial clustering

Y = hierarchy.linkage(X, method = 'average')
plt.figure=(figsize(10,5))
dendrogram = hierarchy.dendrogram(Y)
plt.title('Hierarchical Clustering Dendrogram for Chemical Composition of Ceramics')
plt.xlabel('Chemicals')
plt.show()

# K means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
X['Cluster'] = labels
print(X.head(100))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.drop('Cluster', axis=1))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()