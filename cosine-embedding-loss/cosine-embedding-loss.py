import numpy as np

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    # Write code here
    x1 = np.array(x1)
    x2 = np.array(x2)
    cosine =  np.sum(x1 * x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    if label == 1:
        return 1 - cosine
    elif label == -1:
        return max(0, cosine - margin)
    raise Exception("Invalid label!")
    