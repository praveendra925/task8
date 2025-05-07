import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)
cluster_labels = kmeans.labels_
df['Cluster'] = cluster_labels
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', s=60, edgecolor='k')
plt.title('K-Means Clustering (Iris Dataset)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
print(df.head())
