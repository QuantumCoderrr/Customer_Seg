# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage  

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('../data/Mall_Customers.csv')

# Feature selection
X = data.iloc[:, [3, 4]].values

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
def elbow_method(X):
    elbow = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        elbow.append(kmeans.inertia_)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, 11), elbow, marker='o', linestyle='--')
    plt.xlabel('No. of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method to Determine Optimal K-Value')
    plt.show()

elbow_method(X_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot K-Means clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis', label='Clusters')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title('K-Means Clustering on Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# Compute hierarchical clustering
linked = linkage(X_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(10, 6))
dendro = dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=41, leaf_font_size=10, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Clustered Data Points')
plt.ylabel('Euclidean Distance')
plt.grid(True)
plt.show()

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
clusters = hc.fit_predict(X_scaled)

# Visualize Hierarchical Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="coolwarm", s=120, alpha=0.8, edgecolors="black", marker="o")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Hierarchical Clustering')
plt.legend(title="Clusters")
plt.show()

# Evaluation Metrics
print("K-Means Silhouette Score:", silhouette_score(X_scaled, labels))
print("Hierarchical Clustering Silhouette Score:", silhouette_score(X_scaled, clusters))
