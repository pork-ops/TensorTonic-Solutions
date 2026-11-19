import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    Xt_X_lam = X.T @ X + (lam * np.eye(X.shape[1]))
    return np.linalg.inv(Xt_X_lam) @ X.T @ y 