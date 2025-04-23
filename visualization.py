import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import cm

def plot_silhouette_heatmap(grid_results, title="Silhouette Score Heatmap"):
    """Create heatmap of silhouette scores for different parameters."""
    if 'eps' in grid_results.columns:  # DBSCAN case
        pivot_results = grid_results.pivot(index=['n_components', 'min_samples'], 
                                          columns='eps', 
                                          values='silhouette_score')
        
        plt.figure(figsize=(12, 8))
        for i, (n_components, group) in enumerate(pivot_results.groupby(level=0)):
            plt.subplot(1, len(pivot_results.index.levels[0]), i+1)
            sns.heatmap(group.droplevel(0), annot=True, cmap='viridis', 
                       fmt='.2f', cbar_kws={'label': 'Silhouette Score'})
            plt.title(f"{title} - {n_components} MCA Components")
            plt.xlabel('Epsilon')
            plt.ylabel('Min Samples')
        plt.tight_layout()
    else:  # KMeans, Hierarchical, GMM case
        pivot_results = grid_results.pivot(index='n_components', 
                                          columns='n_clusters', 
                                          values='silhouette_score')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_results, annot=True, cmap='viridis', 
                   fmt='.2f', cbar_kws={'label': 'Silhouette Score'})
        plt.title(title)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Number of MCA Components')
        plt.tight_layout()
    
    return plt.gcf()

def plot_elbow_method(n_clusters_range, inertia_values):
    """Create elbow plot for KMeans inertia."""
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, inertia_values, marker='o', color='b', 
            label='Inertia (K-means Loss)')
    plt.title('K-means Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Loss)')
    plt.grid(True)
    plt.legend()
    
    # Mark 50% reduction threshold
    initial_loss = inertia_values[0]
    half_loss = initial_loss * 0.5
    for i, loss in enumerate(inertia_values):
        if loss <= half_loss:
            plt.axvline(x=n_clusters_range[i], color='r', linestyle='--', 
                      label=f'50% reduction at {n_clusters_range[i]} clusters')
            break
    
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def visualize_tsne_clusters(X_embedded, labels, title="t-SNE Visualization"):
    """Create scatter plot of t-SNE embeddings colored by cluster labels."""
    plt.figure(figsize=(10, 8))
    
    # Set a colormap that works with potentially many clusters
    cmap = plt.cm.get_cmap('viridis', len(np.unique(labels)))
    
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                         c=labels, cmap=cmap, alpha=0.7, s=50)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_true_vs_predicted(X_embedded, y_true, y_pred):
    """Plot true labels vs predicted clusters on same visualization."""
    plt.figure(figsize=(12, 10))
    
    # Create overlay with large points for predicted, small points for true
    # Use different colormap for each to distinguish them
    scatter1 = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                         c=y_pred, cmap='viridis', alpha=0.7, s=70, 
                         edgecolors='w', linewidths=0.5)
    scatter2 = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                         c=y_true, cmap='plasma', alpha=0.4, s=30)
    
    # Add legend to indicate true vs predicted
    plt.legend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, alpha=0.7, label='Predicted Clusters'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=10, alpha=0.4, label='True Labels')
    ], ['Predicted Clusters', 'True Labels'])
    
    plt.title("True Labels vs. Predicted Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_anomaly_scores(scores, anomalies, title="Anomaly Detection Scores"):
    """Create histogram of anomaly detection scores."""
    plt.figure(figsize=(10, 6))
    
    # Get min and max for common x-axis
    min_score, max_score = min(scores), max(scores)
    
    # Create histograms for normal and anomaly points
    plt.hist(scores[~anomalies], bins=50, alpha=0.6, color='blue', 
           label=f'Normal ({sum(~anomalies)} points)', range=(min_score, max_score))
    plt.hist(scores[anomalies], bins=20, alpha=0.8, color='red', 
           label=f'Anomalies ({sum(anomalies)} points)', range=(min_score, max_score))
    
    # Add threshold line - safely calculate the threshold
    is_reversed = np.median(scores[anomalies]) < np.median(scores[~anomalies])
    if is_reversed:
        # For GMM: low scores are anomalies
        threshold = np.mean(scores) - 3*np.std(scores)
    else:
        # For K-means, OCSVM: high scores are anomalies
        threshold = np.mean(scores) + 3*np.std(scores)
    
    plt.axvline(x=threshold, color='black', linestyle='--', 
              label=f'Threshold: {threshold:.2f}')
    
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()