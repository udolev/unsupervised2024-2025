import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (assuming CSV format)
file_path = "lung_cancer_dataset.csv"  # Ensure the file is in your working directory
data = pd.read_csv(file_path)

# Separate features and target label (PULMONARY_DISEASE)
X = data.drop(columns=['PULMONARY_DISEASE'])
y = data['PULMONARY_DISEASE']

# Preprocessing: Select only numerical columns and drop missing values
X_cleaned = X.select_dtypes(include=['float64', 'int64']).dropna()

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# Encode the target column to numerical values (for visualization of true labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply TSNE for 2D visualization
tsne_2d = TSNE(n_components=2, random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

# Apply TSNE for 3D visualization
tsne_3d = TSNE(n_components=3, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

# Apply K-Means clustering (using 3 clusters as an example)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Apply Gaussian Mixture Model clustering (using 3 components)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualization helper function for 2D with saving option
def plot_clusters_2d(data, labels, title, filename):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette="viridis", s=10)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title="Cluster", loc='best', bbox_to_anchor=(1, 1))
    plt.savefig(filename, bbox_inches='tight')  # Save the plot to a file
    plt.show()

# Visualization helper function for 3D with saving option
def plot_clusters_3d(data, labels, title, filename):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=10)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc='best', bbox_to_anchor=(1, 1))
    ax.add_artist(legend)
    plt.savefig(filename, bbox_inches='tight')  # Save the plot to a file
    plt.show()

# Visualize TSNE (2D) with true labels (encoded)
plot_clusters_2d(X_tsne_2d, y_encoded, "t-SNE Visualization (2D) with True Labels", "tsne_2d_true_labels.png")

# Visualize TSNE (2D) with K-Means clusters
plot_clusters_2d(X_tsne_2d, kmeans_labels, "t-SNE Visualization (2D) with K-Means Clusters", "tsne_2d_kmeans.png")

# Visualize TSNE (2D) with GMM clusters
plot_clusters_2d(X_tsne_2d, gmm_labels, "t-SNE Visualization (2D) with GMM Clusters", "tsne_2d_gmm.png")

# Visualize TSNE (2D) with DBSCAN clusters
plot_clusters_2d(X_tsne_2d, dbscan_labels, "t-SNE Visualization (2D) with DBSCAN Clusters", "tsne_2d_dbscan.png")

# Visualize TSNE (3D) with true labels (encoded)
plot_clusters_3d(X_tsne_3d, y_encoded, "t-SNE Visualization (3D) with True Labels", "tsne_3d_true_labels.png")

# Visualize TSNE (3D) with K-Means clusters
plot_clusters_3d(X_tsne_3d, kmeans_labels, "t-SNE Visualization (3D) with K-Means Clusters", "tsne_3d_kmeans.png")

# Visualize TSNE (3D) with GMM clusters
plot_clusters_3d(X_tsne_3d, gmm_labels, "t-SNE Visualization (3D) with GMM Clusters", "tsne_3d_gmm.png")

# Visualize TSNE (3D) with DBSCAN clusters
plot_clusters_3d(X_tsne_3d, dbscan_labels, "t-SNE Visualization (3D) with DBSCAN Clusters", "tsne_3d_dbscan.png")
