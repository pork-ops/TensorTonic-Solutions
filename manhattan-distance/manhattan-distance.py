import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    return float(np.sum(np.abs(np.array(x) - np.array(y))))