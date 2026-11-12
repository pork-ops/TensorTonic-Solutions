import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    # Write code here
    diagonal = []
    for i in range(len(A)):
        diagonal.append(A[i][i])
    return np.sum(np.array(diagonal))