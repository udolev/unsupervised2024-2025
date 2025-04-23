import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def kmeans_anomaly_detection(X, n_clusters):
    """Detect anomalies using K-means centroids."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Compute distances to assigned centroids
    distances = np.sqrt(np.sum((X - kmeans.cluster_centers_[labels])**2, axis=1))
    
    # Define anomalies as points with distance > mean + 3*std
    threshold = np.mean(distances) + 3 * np.std(distances)
    anomalies = distances > threshold
    
    return anomalies, distances

def gmm_anomaly_detection(X, n_components):
    """Detect anomalies using Gaussian Mixture Models."""
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Get log-likelihood scores
    scores = gmm.score_samples(X)
    
    # Define anomalies as points with log-likelihood < mean - 3*std
    threshold = np.mean(scores) - 3 * np.std(scores)
    anomalies = scores < threshold
    
    return anomalies, scores

def one_class_svm_anomaly_detection(X, nu=0.01):
    """Detect anomalies using One-Class SVM."""
    svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    svm.fit(X)
    
    # Get decision function scores
    scores = svm.decision_function(X)
    
    # SVM already returns -1 for anomalies, 1 for normal
    anomalies = svm.predict(X) == -1
    
    return anomalies, scores