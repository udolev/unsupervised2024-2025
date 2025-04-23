import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, mutual_info_score
from scipy import stats

def calculate_silhouette(X, labels):
    """Calculate silhouette score for a clustering."""
    if len(set(labels)) < 2:
        return np.nan
    return silhouette_score(X, labels)

def calculate_mutual_info(labels, external_var):
    """Calculate mutual information between cluster labels and external variable."""
    return mutual_info_score(labels, external_var)

def compare_algorithms_anova(silhouette_scores):
    """Compare clustering algorithms using ANOVA on silhouette scores."""
    # Filter out algorithms with NaN values
    valid_scores = {algo: scores for algo, scores in silhouette_scores.items() 
                   if not np.isnan(scores).any()}
    
    if len(valid_scores) < 2:
        return np.nan, np.nan
    
    values = list(valid_scores.values())
    f_statistic, p_value = stats.f_oneway(*values)
    return f_statistic, p_value

def compare_best_algorithms_ttest(algo1_scores, algo2_scores):
    """Compare two best algorithms using paired t-test."""
    # Remove NaN values if any
    valid_indices = ~(np.isnan(algo1_scores) | np.isnan(algo2_scores))
    if sum(valid_indices) < 2:
        return np.nan, np.nan
    
    a1_scores = np.array(algo1_scores)[valid_indices]
    a2_scores = np.array(algo2_scores)[valid_indices]
    
    t_statistic, p_value = stats.ttest_rel(a1_scores, a2_scores)
    return t_statistic, p_value

def find_optimal_clusters(inertia_values, n_clusters_range):
    """Find optimal number of clusters using elbow method with 50% threshold."""
    initial_inertia = inertia_values[0]
    reductions = [(initial_inertia - inertia) / initial_inertia * 100 
                  for inertia in inertia_values]
    
    # Find where reduction is at least 50%
    for i, reduction in enumerate(reductions):
        if reduction >= 50:
            return n_clusters_range[i], reduction
    
    # If no 50% reduction, return max reduction point
    max_idx = np.argmax(reductions)
    return n_clusters_range[max_idx], reductions[max_idx]