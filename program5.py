import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(df)
df['Cluster'] = cluster_labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', s=60, edgecolor='k')
plt.title("K-Means Clustering with Color Coding")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
legend_labels = [f'Cluster {i}' for i in range(kmeans.n_clusters)]
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
           for i in range(kmeans.n_clusters)]
plt.legend(handles, legend_labels, title="Clusters")
plt.show()
score = silhouette_score(df.drop(columns='Cluster'), cluster_labels)
print(f"Silhouette Score: {score:.3f}")
