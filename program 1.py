import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Optional: apply PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='gray', edgecolor='k', s=60)
plt.title('PCA Projection of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
