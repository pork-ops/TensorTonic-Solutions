import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    # Write code here
    x = np.array(x)
    q = np.array(q)
    positions = q / 100.0 * (x.size - 1)
    x = np.sort(x)
    l = np.floor(positions).astype(int)
    u = np.ceil(positions).astype(int)
    w = positions - l
    x_u = x[u]
    x_l = x[l]
    return ((1 - w) * x_l) + (w * x_u)