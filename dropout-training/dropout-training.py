import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.array(x)
    z = np.zeros(x.shape)
    s = np.ndarray(x.shape)
    s.fill(1/(1-p))
    
    if rng is None:
        rng = np.random.random(x.shape)
    else:
        rng = rng.random(x.shape)
    dropout_pattern = np.where(rng > (1 - p), z, s)
    output = x * dropout_pattern

    return (output, dropout_pattern)