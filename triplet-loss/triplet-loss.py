import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    anchor = np.atleast_2d(np.array(anchor))
    positive = np.atleast_2d(np.array(positive))
    negative = np.atleast_2d(np.array(negative))
    d_a_p = np.sum(np.pow(anchor - positive, 2), axis=1) 
    d_a_n = np.sum(np.pow(anchor - negative, 2), axis=1) 
    pre_loss = d_a_p - d_a_n + margin
    loss = np.where(pre_loss > 0, pre_loss, 0)
    return float(loss.mean())