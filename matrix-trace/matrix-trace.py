import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    # Write code here
    indexes = np.arange(0, len(A))
    A = np.array(A)
    return float(np.sum(A[indexes,indexes]))