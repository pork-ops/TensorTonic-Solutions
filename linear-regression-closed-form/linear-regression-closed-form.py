import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    Xt_X = X.T @ X
    inverse = np.linalg.inv(Xt_X)
    Xt_y = X.T @ y
    return inverse @ Xt_y