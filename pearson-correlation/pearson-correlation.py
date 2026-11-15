import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.array(X)
    means = np.mean(X, axis=0)
    centered = X - means
    cov = centered.T @ centered / (X.shape[0] - 1)
    stds = np.sqrt(np.diagonal(cov))
    correlation = cov / np.outer(stds, stds)
    return correlation