import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    PE   = np.empty((seq_len, d_model), dtype=float)
    i,j = np.indices((seq_len, round((d_model/2)+0.01)))
    aux = i/np.power(base, 2*j/d_model)
    PE[:,  ::2] = np.sin(aux)
    PE[:, 1::2] = np.cos(aux)[:, :round((d_model/2)-0.01)]
    return PE