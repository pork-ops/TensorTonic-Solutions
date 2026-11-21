import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    items = margin - y_true * y_score
    items = np.where(items > 0, items, 0)
    if reduction == "mean":
        return float(np.mean(items))
    if reduction == "sum":
        return float(np.sum(items))
    raise Exception("Invalid reduction type!")