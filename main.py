import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler

# Import from project modules
from config import (RANDOM_SEED, NUM_CV_RUNS, CATEGORICAL_COLS, 
                   NUMERICAL_COLS, TARGET_COL, CLUSTER_RANGE, MCA_DIMENSIONS, 
                   DBSCAN_EPS_RANGE, DBSCAN_MIN_SAMPLES_RANGE, OUTPUT_DIR)
from data_loader import load_data, preprocess_data, apply_mca, combine_features
from dimensionality import apply_tsne
from clustering import (kmeans_clustering, hierarchical_clustering, 
                       dbscan_clustering, gmm_clustering, grid_search_kmeans,
                       grid_search_hierarchical, grid_search_dbscan, grid_search_gmm)
from anomaly_detection import (kmeans_anomaly_detection, gmm_anomaly_detection, 
                              one_class_svm_anomaly_detection)
from evaluation import (calculate_silhouette, calculate_mutual_info, 
                       compare_algorithms_anova, compare_best_algorithms_ttest,
                       find_optimal_clusters)
from visualization import (plot_silhouette_heatmap, plot_elbow_method,
                         visualize_tsne_clusters, plot_true_vs_predicted,
                         plot_anomaly_scores)

# Set random seed
np.random.seed(RANDOM_SEED)

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('lung_cancer_dataset.csv')
    X_cat, X_num_scaled, y = preprocess_data(df)
    
    # 2. Grid search for optimal parameters for each algorithm
    print("Running grid search for clustering algorithms...")
    
    # KMeans
    kmeans_results = grid_search_kmeans(X_cat, CLUSTER_RANGE, MCA_DIMENSIONS)
    best_kmeans = kmeans_results.loc[kmeans_results['silhouette_score'].idxmax()]
    print(f"Best KMeans: {int(best_kmeans['n_components'])} components, {int(best_kmeans['n_clusters'])} clusters")
    
    # Hierarchical
    hierarchical_results = grid_search_hierarchical(X_cat, CLUSTER_RANGE, MCA_DIMENSIONS)
    best_hierarchical = hierarchical_results.loc[hierarchical_results['silhouette_score'].idxmax()]
    print(f"Best Hierarchical: {int(best_hierarchical['n_components'])} components, {int(best_hierarchical['n_clusters'])} clusters")
    
    # DBSCAN
    dbscan_results = grid_search_dbscan(X_cat, DBSCAN_EPS_RANGE, DBSCAN_MIN_SAMPLES_RANGE, MCA_DIMENSIONS)
    best_dbscan = dbscan_results.loc[dbscan_results['silhouette_score'].idxmax()]
    print(f"Best DBSCAN: {int(best_dbscan['n_components'])} components, eps={best_dbscan['eps']:.2f}, min_samples={int(best_dbscan['min_samples'])}")
    
    # GMM
    gmm_results = grid_search_gmm(X_cat, CLUSTER_RANGE, MCA_DIMENSIONS)
    best_gmm = gmm_results.loc[gmm_results['silhouette_score'].idxmax()]
    print(f"Best GMM: {int(best_gmm['n_components'])} components, {int(best_gmm['n_clusters'])} clusters")
    
    # Save heatmaps for each algorithm
    plot_silhouette_heatmap(kmeans_results, "KMeans Silhouette Score").savefig(
        os.path.join(OUTPUT_DIR, 'kmeans_heatmap.png'))
    plot_silhouette_heatmap(hierarchical_results, "Hierarchical Silhouette Score").savefig(
        os.path.join(OUTPUT_DIR, 'hierarchical_heatmap.png'))
    plot_silhouette_heatmap(gmm_results, "GMM Silhouette Score").savefig(
        os.path.join(OUTPUT_DIR, 'gmm_heatmap.png'))
    
    # 3. Cross-validation to compare algorithms
    print("Performing cross-validation for algorithm comparison...")
    
    # Store silhouette scores for each algorithm across CV runs
    silhouette_scores = defaultdict(list)
    
    # Track best algorithm across CV runs
    for cv in range(NUM_CV_RUNS):
        print(f"Cross-validation run {cv+1}/{NUM_CV_RUNS}")
        
        # Apply each algorithm with its best parameters
        # KMeans
        kmeans_mca = apply_mca(X_cat, int(best_kmeans['n_components']))
        kmeans_X = combine_features(kmeans_mca, X_num_scaled)
        kmeans_labels, _, _ = kmeans_clustering(kmeans_X, int(best_kmeans['n_clusters']))
        silhouette_scores['kmeans'].append(calculate_silhouette(kmeans_X, kmeans_labels))
        
        # Hierarchical
        hierarchical_mca = apply_mca(X_cat, int(best_hierarchical['n_components']))
        hierarchical_X = combine_features(hierarchical_mca, X_num_scaled)
        hierarchical_labels = hierarchical_clustering(hierarchical_X, int(best_hierarchical['n_clusters']))
        silhouette_scores['hierarchical'].append(calculate_silhouette(hierarchical_X, hierarchical_labels))
        
        # DBSCAN
        dbscan_mca = apply_mca(X_cat, int(best_dbscan['n_components']))
        dbscan_X = combine_features(dbscan_mca, X_num_scaled)
        dbscan_labels, _, _ = dbscan_clustering(dbscan_X, best_dbscan['eps'], int(best_dbscan['min_samples']))
        if len(set(dbscan_labels)) > 1:
            silhouette_scores['dbscan'].append(calculate_silhouette(dbscan_X, dbscan_labels))
        
        # GMM
        gmm_mca = apply_mca(X_cat, int(best_gmm['n_components']))
        gmm_X = combine_features(gmm_mca, X_num_scaled)
        gmm_labels = gmm_clustering(gmm_X, int(best_gmm['n_clusters']))
        silhouette_scores['gmm'].append(calculate_silhouette(gmm_X, gmm_labels))
    
    # 4. Statistical tests to compare algorithms
    print("Performing statistical tests...")
    
    # ANOVA test
    f_statistic, p_value = compare_algorithms_anova(silhouette_scores)
    print(f"ANOVA test: F={f_statistic:.4f}, p={p_value:.6f}")
    
    # Find best two algorithms
    algo_means = {algo: np.mean(scores) for algo, scores in silhouette_scores.items()}
    top_algos = sorted(algo_means.items(), key=lambda x: x[1], reverse=True)[:2]
    best_algo, second_best_algo = top_algos[0][0], top_algos[1][0]
    
    # Paired t-test between best two algorithms
    t_statistic, t_p_value = compare_best_algorithms_ttest(
        silhouette_scores[best_algo], silhouette_scores[second_best_algo])
    print(f"Paired t-test ({best_algo} vs {second_best_algo}): t={t_statistic:.4f}, p={t_p_value:.6f}")
    
    # 5. Apply best algorithm and compute elbow method
    best_algo_params = locals()[f"best_{best_algo}"]
    
    if best_algo == 'dbscan':
        best_mca = apply_mca(X_cat, int(best_algo_params['n_components']))
        X_combined = combine_features(best_mca, X_num_scaled)
        labels, n_clusters, n_noise = dbscan_clustering(
            X_combined, best_algo_params['eps'], int(best_algo_params['min_samples']))
    else:
        # Elbow method (only for KMeans)
        if best_algo == 'kmeans':
            inertia_values = []
            best_mca = apply_mca(X_cat, int(best_algo_params['n_components']))
            X_combined = combine_features(best_mca, X_num_scaled)
            
            for n_clusters in CLUSTER_RANGE:
                _, inertia, _ = kmeans_clustering(X_combined, n_clusters)
                inertia_values.append(inertia)
            
            optimal_clusters, reduction = find_optimal_clusters(inertia_values, CLUSTER_RANGE)
            print(f"Optimal clusters (50% reduction): {optimal_clusters}")
            
            # Save elbow plot
            elbow_plot = plot_elbow_method(CLUSTER_RANGE, inertia_values)
            elbow_plot.savefig(os.path.join(OUTPUT_DIR, 'kmeans_elbow.png'))
            
            # Apply kmeans with optimal clusters
            labels, _, _ = kmeans_clustering(X_combined, optimal_clusters)
        else:
            best_mca = apply_mca(X_cat, int(best_algo_params['n_components']))
            X_combined = combine_features(best_mca, X_num_scaled)
            if best_algo == 'hierarchical':
                labels = hierarchical_clustering(X_combined, int(best_algo_params['n_clusters']))
            else:  # GMM
                labels = gmm_clustering(X_combined, int(best_algo_params['n_clusters']))
    
    # 6. External Variable Association
    print("Calculating mutual information with target...")
    mi_score = calculate_mutual_info(labels, y)
    print(f"Mutual information with target: {mi_score:.4f}")
    
    # For reference, also check mutual information for other algorithms
    external_mi = {
        'kmeans': calculate_mutual_info(kmeans_labels, y),
        'hierarchical': calculate_mutual_info(hierarchical_labels, y),
        'gmm': calculate_mutual_info(gmm_labels, y)
    }
    if len(set(dbscan_labels)) > 1:
        external_mi['dbscan'] = calculate_mutual_info(dbscan_labels, y)
        
    print("Mutual information with target for all algorithms:")
    for algo, score in external_mi.items():
        print(f"  {algo}: {score:.4f}")
    
    # 7. Anomaly Detection
    print("Performing anomaly detection...")
    
    # K-means anomaly detection
    kmeans_anomalies, kmeans_scores = kmeans_anomaly_detection(X_combined, int(best_kmeans['n_clusters']))
    plot_anomaly_scores(kmeans_scores, kmeans_anomalies, "K-means Anomaly Scores").savefig(
        os.path.join(OUTPUT_DIR, 'kmeans_anomalies.png'))
    print(f"K-means detected {sum(kmeans_anomalies)} anomalies ({sum(kmeans_anomalies)/len(kmeans_anomalies)*100:.2f}%)")
    
    # GMM anomaly detection
    gmm_anomalies, gmm_scores = gmm_anomaly_detection(X_combined, int(best_gmm['n_clusters']))
    plot_anomaly_scores(gmm_scores, gmm_anomalies, "GMM Anomaly Scores").savefig(
        os.path.join(OUTPUT_DIR, 'gmm_anomalies.png'))
    print(f"GMM detected {sum(gmm_anomalies)} anomalies ({sum(gmm_anomalies)/len(gmm_anomalies)*100:.2f}%)")
    
    # One-class SVM anomaly detection
    ocsvm_anomalies, ocsvm_scores = one_class_svm_anomaly_detection(X_combined)
    plot_anomaly_scores(ocsvm_scores, ocsvm_anomalies, "One-Class SVM Anomaly Scores").savefig(
        os.path.join(OUTPUT_DIR, 'ocsvm_anomalies.png'))
    print(f"One-class SVM detected {sum(ocsvm_anomalies)} anomalies ({sum(ocsvm_anomalies)/len(ocsvm_anomalies)*100:.2f}%)")
    
    # Analyze relation between anomalies and target
    anomaly_mi = {
        'kmeans': calculate_mutual_info(kmeans_anomalies, y),
        'gmm': calculate_mutual_info(gmm_anomalies, y),
        'ocsvm': calculate_mutual_info(ocsvm_anomalies, y)
    }
    print("Mutual information between anomalies and target:")
    for algo, score in anomaly_mi.items():
        print(f"  {algo}: {score:.4f}")
    
    # 8. Create t-SNE visualization
    print("Creating visualizations...")
    X_tsne = apply_tsne(X_combined)
    
    # Save cluster visualization
    tsne_plot = visualize_tsne_clusters(X_tsne, labels, f"Best Algorithm ({best_algo}) Clusters")
    tsne_plot.savefig(os.path.join(OUTPUT_DIR, f'{best_algo}_clusters.png'))
    
    # Save true vs predicted visualization
    comparison_plot = plot_true_vs_predicted(X_tsne, y, labels)
    comparison_plot.savefig(os.path.join(OUTPUT_DIR, 'true_vs_predicted.png'))
    
    # 9. Dive deeper into clusters to understand medical implications
    print("\nAnalyzing cluster characteristics...")
    
    # Get the best clustering results
    best_algo_name = best_algo  # The algorithm name with highest score
    
    # Use K-means for interpretability even if it's not the best
    # (Medical interpretations are easier with a centroid-based method)
    interpretation_mca = apply_mca(X_cat, int(best_kmeans['n_components']))
    interpretation_X = combine_features(interpretation_mca, X_num_scaled)
    _, _, kmeans_model = kmeans_clustering(interpretation_X, int(best_kmeans['n_clusters']))
    
    # Combine original data with cluster labels
    cluster_df = df.copy()
    cluster_df['cluster'] = kmeans_model.labels_
    
    # Analyze each cluster's characteristics
    cluster_stats = []
    for cluster_id in range(int(best_kmeans['n_clusters'])):
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        
        # Calculate mean values for numerical features
        numeric_means = cluster_data[NUMERICAL_COLS].mean()
        
        # Calculate most common values for categorical features
        categorical_modes = {}
        for col in CATEGORICAL_COLS:
            mode_value = cluster_data[col].mode()[0]
            mode_pct = (cluster_data[col] == mode_value).mean() * 100
            categorical_modes[col] = (mode_value, mode_pct)
        
        # Calculate pulmonary disease percentage
        disease_pct = (cluster_data[TARGET_COL] == 'YES').mean() * 100
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'size_pct': len(cluster_data) / len(cluster_df) * 100,
            'disease_pct': disease_pct,
            'numeric_means': numeric_means,
            'categorical_modes': categorical_modes
        })
    
    # Output medical/biological insights
    print("\n=== CLUSTER ANALYSIS FOR MEDICAL INSIGHTS ===")
    for cluster in cluster_stats:
        print(f"\nCluster {cluster['cluster_id']} ({cluster['size']} patients, {cluster['size_pct']:.1f}% of total)")
        print(f"Disease rate: {cluster['disease_pct']:.1f}% have pulmonary disease")
        
        print("\nKey characteristics:")
        # Get top 3 distinctive numerical features
        for col in NUMERICAL_COLS:
            overall_mean = df[col].mean()
            cluster_mean = cluster['numeric_means'][col]
            diff_pct = ((cluster_mean - overall_mean) / overall_mean) * 100
            print(f"- {col}: {cluster_mean:.1f} ({diff_pct:+.1f}% vs. population mean)")
        
        # Get top 3 distinctive categorical features
        for col, (mode, pct) in list(cluster['categorical_modes'].items())[:5]:
            print(f"- {col}: {mode} ({pct:.1f}% of cluster)")
    
    # Create a radar chart of cluster profiles
    features_to_plot = ['AGE', 'ENERGY_LEVEL', 'OXYGEN_SATURATION']
    
    # Standardize features for radar chart
    scaler = StandardScaler()
    df_std = pd.DataFrame(
        scaler.fit_transform(df[features_to_plot]),
        columns=features_to_plot
    )
    df_std['cluster'] = kmeans_model.labels_
    
    # Compute cluster means
    cluster_means = df_std.groupby('cluster').mean()
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 8))
    
    # Number of variables
    categories = features_to_plot
    N = len(categories)
    
    # Create angles for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "1", "2"], color="grey", size=10)
    plt.ylim(-2, 2)
    
    # Plot each cluster
    cmap = cm.get_cmap('viridis', len(cluster_means))
    for i, (idx, values) in enumerate(cluster_means.iterrows()):
        values_list = values.tolist()
        values_list += values_list[:1]  # Close the loop
        ax.plot(angles, values_list, linewidth=2, linestyle='solid', 
               color=cmap(i), label=f"Cluster {idx}")
        ax.fill(angles, values_list, color=cmap(i), alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Cluster Profiles Across Key Health Indicators", size=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_profiles_radar.png'))
    
    # 10. Analyze anomalies in relation to clusters and disease
    anomaly_analysis = pd.DataFrame({
        'kmeans_anomaly': kmeans_anomalies,
        'gmm_anomaly': gmm_anomalies,
        'ocsvm_anomaly': ocsvm_anomalies,
        'cluster': labels,
        'disease': y
    })
    
    # Disease rate among anomalies vs normal points
    print("\n=== ANOMALY ANALYSIS ===")
    for method in ['kmeans', 'gmm', 'ocsvm']:
        anomaly_col = f'{method}_anomaly'
        disease_rate_anomaly = anomaly_analysis[anomaly_analysis[anomaly_col]]['disease'].mean() * 100
        disease_rate_normal = anomaly_analysis[~anomaly_analysis[anomaly_col]]['disease'].mean() * 100
        
        print(f"\n{method.upper()} Anomalies:")
        print(f"Disease rate in anomalies: {disease_rate_anomaly:.1f}%")
        print(f"Disease rate in normal points: {disease_rate_normal:.1f}%")
        print(f"Difference: {disease_rate_anomaly - disease_rate_normal:+.1f}%")
        
        # Cluster distribution of anomalies
        anomaly_clusters = anomaly_analysis[anomaly_analysis[anomaly_col]]['cluster'].value_counts(normalize=True) * 100
        print("\nCluster distribution of anomalies:")
        for cluster, pct in anomaly_clusters.items():
            print(f"- Cluster {cluster}: {pct:.1f}%")
    
    print("\n=== CONCLUSIONS ===")
    print("Based on clustering and anomaly detection analysis:")
    print("1. The dataset appears to separate into distinct patient groups with different characteristics")
    print("2. Anomaly detection reveals potential outlier cases that may represent misdiagnoses or unusual presentations")
    print("3. The relationship between clustering patterns and pulmonary disease suggests...")
    print("   (This would be filled in with specific medical insights based on the results)")
    
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()