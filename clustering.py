import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def kmeans_clustering(X, n_clusters, random_state=42):
    """Apply KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.inertia_, kmeans

def hierarchical_clustering(X, n_clusters):
    """Apply Agglomerative Hierarchical clustering."""
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X)
    return labels

def dbscan_clustering(X, eps, min_samples):
    """Apply DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    n_noise = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters, n_noise

def gmm_clustering(X, n_components, random_state=42):
    """Apply Gaussian Mixture Model clustering."""
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X)
    return labels

def grid_search_kmeans(X, n_clusters_range, mca_dims):
    """Perform grid search for KMeans with different clusters and MCA dimensions."""
    results = []
    
    for n_components in mca_dims:
        mca = prince.MCA(n_components=n_components)
        df_mca = mca.fit_transform(X)
        
        for n_clusters in n_clusters_range:
            labels, _, _ = kmeans_clustering(df_mca, n_clusters)
            sil_score = silhouette_score(df_mca, labels)
            results.append((n_components, n_clusters, sil_score))
    
    return pd.DataFrame(results, columns=['n_components', 'n_clusters', 'silhouette_score'])

# Similarly implement grid_search functions for other algorithms
def grid_search_hierarchical(X, n_clusters_range, mca_dims):
    """Perform grid search for Hierarchical clustering with different clusters and MCA dimensions."""
    results = []
    
    for n_components in mca_dims:
        mca = prince.MCA(n_components=n_components)
        df_mca = mca.fit_transform(X)
        
        for n_clusters in n_clusters_range:
            labels = hierarchical_clustering(df_mca, n_clusters)
            sil_score = silhouette_score(df_mca, labels)
            results.append((n_components, n_clusters, sil_score))
    
    return pd.DataFrame(results, columns=['n_components', 'n_clusters', 'silhouette_score'])

def grid_search_dbscan(X, eps_range, min_samples_range, mca_dims):
    """Perform grid search for DBSCAN with different eps, min_samples and MCA dimensions."""
    results = []
    
    for n_components in mca_dims:
        mca = prince.MCA(n_components=n_components)
        df_mca = mca.fit_transform(X)
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                labels, n_clusters, n_noise = dbscan_clustering(df_mca, eps, min_samples)
                sil_score = silhouette_score(df_mca, labels) if n_clusters > 1 else -1
                results.append((n_components, eps, min_samples, n_clusters, n_noise, sil_score))
    
    return pd.DataFrame(results, columns=['n_components', 'eps', 'min_samples', 'n_clusters', 'n_noise', 'silhouette_score'])

def grid_search_gmm(X, n_components_range, mca_dims):
    """Perform grid search for GMM with different components and MCA dimensions."""
    results = []
    
    for n_components in mca_dims:
        mca = prince.MCA(n_components=n_components)
        df_mca = mca.fit_transform(X)
        
        for n_clusters in n_components_range:
            labels = gmm_clustering(df_mca, n_clusters)
            sil_score = silhouette_score(df_mca, labels)
            results.append((n_components, n_clusters, sil_score))
    
    return pd.DataFrame(results, columns=['n_components', 'n_clusters', 'silhouette_score'])