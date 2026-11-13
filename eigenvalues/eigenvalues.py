import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    # Write code here
    eigen = np.linalg.eigvals(matrix)
    return np.sort(eigen.real)
    