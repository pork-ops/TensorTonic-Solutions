import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X = np.array(X)
    mean = np.mean(X, axis=0)
    centered = X - mean
    return (centered.T @ centered) / (centered.shape[0] - 1)