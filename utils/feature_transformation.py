from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def normalize(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, scaler


def apply_pca(features, variance_threshold=0.95):
    """
    Apply PCA to reduce dimensionality while retaining the specified variance.
    """
    pca = PCA(n_components=variance_threshold)
    reduced_features = pca.fit_transform(features)
    print(f"Reduced from {features.shape[1]} to {reduced_features.shape[1]} dimensions")
    return reduced_features, pca


def remove_low_variance_features(features, threshold=0.01):
    """
    Removes features with variance below the threshold.
    """
    selector = VarianceThreshold(threshold)
    reduced_features = selector.fit_transform(features)
    print(f"Reduced from {features.shape[1]} to {reduced_features.shape[1]} dimensions")
    return reduced_features, selector
