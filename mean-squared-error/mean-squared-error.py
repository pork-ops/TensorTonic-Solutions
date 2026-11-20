import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.sum(np.pow(y_pred - y_true, 2)) / y_pred.shape[0]
     