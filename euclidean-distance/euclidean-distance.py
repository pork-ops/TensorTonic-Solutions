import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    # Write code here
    X = np.array(x)
    Y = np.array(y)
    return float(np.sqrt(np.sum(np.power(X-Y,2))))