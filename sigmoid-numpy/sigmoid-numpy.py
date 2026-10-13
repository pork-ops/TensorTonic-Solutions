import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.array(x)
    x = np.exp(-1*x)
    x = 1 / (1 + x)
    return x