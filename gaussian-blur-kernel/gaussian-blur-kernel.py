import numpy as np

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # Write code here
    base = np.pow(np.arange(size) - size // 2, 2)
    y = np.tile(base, (size, 1))
    x = y.T
    g = np.exp(-( (x + y) / (2 * np.pow(sigma,2))))
    total = np.sum(g)
    return (g / total).tolist() # Normalize