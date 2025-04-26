import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from config import NUM_SAMPLES

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

def combine_features(X_cat_mca, X_num_scaled):
    """Combine MCA-transformed categorical features with scaled numerical features."""
    X_combined = pd.concat([X_cat_mca.reset_index(drop=True), 
                           pd.DataFrame(X_num_scaled).reset_index(drop=True)], axis=1)
    # Ensure all column names are strings for compatibility with sklearn
    X_combined.columns = X_combined.columns.astype(str)
    # Scale the combined features
    X_combined = StandardScaler().fit_transform(X_combined)
    return X_combined

def grid_search_kmeans(X_cat, X_num_scaled, n_clusters_range, mca_dims):
    """Perform grid search for KMeans with different clusters and MCA dimensions on combined features."""
    results = []
    
    for n_components in mca_dims:
        # Apply MCA with the current number of components
        mca = prince.MCA(n_components=n_components)
        X_cat_mca = mca.fit_transform(X_cat)
        
        # Combine with numerical features
        X_combined = combine_features(X_cat_mca, X_num_scaled)
        
        for n_clusters in n_clusters_range:
            # Fit KMeans on combined data
            labels, inertia, _ = kmeans_clustering(X_combined, n_clusters)
            cluster_counts = np.bincount(labels)
            sil_score = silhouette_score(X_combined, labels)
            results.append((n_components, n_clusters, sil_score, inertia, cluster_counts))
    
    return pd.DataFrame(results, columns=['n_components', 'n_clusters', 'silhouette_score', 'inertia', 'cluster_counts'])

def grid_search_hierarchical(X_cat, X_num_scaled, n_clusters_range, mca_dims):
    """Perform grid search for Hierarchical clustering with different clusters and MCA dimensions on combined features."""
    results = []
    
    for n_components in mca_dims:
        # Apply MCA with the current number of components
        mca = prince.MCA(n_components=n_components)
        X_cat_mca = mca.fit_transform(X_cat)
        
        # Combine with numerical features
        X_combined = combine_features(X_cat_mca, X_num_scaled)
        
        for n_clusters in n_clusters_range:
            # Fit Hierarchical clustering on combined data
            labels = hierarchical_clustering(X_combined, n_clusters)
            cluster_counts = np.bincount(labels)
            sil_score = silhouette_score(X_combined, labels)
            results.append((n_components, n_clusters, sil_score, cluster_counts))
    
    return pd.DataFrame(results, columns=['n_components', 'n_clusters', 'silhouette_score', 'cluster_counts'])

def grid_search_dbscan(X_cat, X_num_scaled, eps_range, min_samples_range, mca_dims):
    """Perform grid search for DBSCAN with different eps, min_samples and MCA dimensions on combined features."""
    results = []
    
    for n_components in mca_dims:
        # Apply MCA with the current number of components
        mca = prince.MCA(n_components=n_components)
        X_cat_mca = mca.fit_transform(X_cat)
        
        # Combine with numerical features
        X_combined = combine_features(X_cat_mca, X_num_scaled)
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                # Fit DBSCAN on combined data
                labels, n_clusters, n_noise = dbscan_clustering(X_combined, eps, min_samples)
                cluster_counts = np.bincount(labels[labels != -1])
                
                # Only calculate silhouette if there are more than 1 cluster
                if n_clusters > 1:
                    # Exclude noise points for silhouette calculation
                    non_noise_mask = labels != -1
                    if sum(non_noise_mask) > NUM_SAMPLES*0.9:
                        sil_score = silhouette_score(X_combined[non_noise_mask], labels[non_noise_mask])
                    else:
                        sil_score = -1
                else:
                    sil_score = -1
                
                results.append((n_components, eps, min_samples, n_clusters, n_noise, sil_score, cluster_counts))
    
    return pd.DataFrame(results, columns=['n_components', 'eps', 'min_samples', 'n_clusters', 'n_noise', 'silhouette_score', 'cluster_counts'])

def grid_search_gmm(X_cat, X_num_scaled, n_components_range, mca_dims):
    """Perform grid search for GMM with different components and MCA dimensions on combined features."""
    results = []
    
    for n_components in mca_dims:
        # Apply MCA with the current number of components
        mca = prince.MCA(n_components=n_components)
        X_cat_mca = mca.fit_transform(X_cat)
        
        # Combine with numerical features
        X_combined = combine_features(X_cat_mca, X_num_scaled)
        
        for n_clusters in n_components_range:
            # Fit GMM on combined data
            labels = gmm_clustering(X_combined, n_clusters)
            cluster_counts = np.bincount(labels)
            sil_score = silhouette_score(X_combined, labels)
            results.append((n_components, n_clusters, sil_score, cluster_counts))
    
    return pd.DataFrame(results, columns=['n_components', 'n_clusters', 'silhouette_score', 'cluster_counts'])