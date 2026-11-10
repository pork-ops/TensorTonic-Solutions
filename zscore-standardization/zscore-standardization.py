import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    # Write code here
    mean = np.mean(X, axis, keepdims=True) 
    std = np.std(X, axis, keepdims=True)
    std = np.where(std > eps, std, np.inf)
    return (X - mean) / std