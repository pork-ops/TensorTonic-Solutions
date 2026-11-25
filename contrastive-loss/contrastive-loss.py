import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    diff = (a - b).reshape(-1, a.shape[-1])
    d = np.linalg.norm(diff, axis=1)
    l = y * np.pow(d,2) + (1 - y) * np.pow(np.maximum(0, margin - d),2)

    if reduction == "mean":
        return float(l.mean())
    elif reduction == "sum":
        return float(l.sum())
    raise Exception("Invalid reduction!")