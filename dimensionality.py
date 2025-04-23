from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def apply_tsne(X, n_components=2, perplexity=30, random_state=42):
    """Apply t-SNE for visualization."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state)
    return tsne.fit_transform(X)

def apply_pca(X, n_components=2, random_state=42):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)