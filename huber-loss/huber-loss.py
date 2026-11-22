import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    e = y_true - y_pred
    L = np.where(np.abs(e) > delta, delta * (np.abs(e) - ((1/2) * delta)), (1/2) * np.pow(e, 2))
    return float(np.mean(L))