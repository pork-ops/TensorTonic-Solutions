import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    # This would be interesting https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    y = np.array(y)
    counts = np.array(np.unique(y, return_counts=True))[1,:]
    total = np.sum(counts)
    p = counts / total
    h = -np.sum(p*np.log2(p))
    return float(h)