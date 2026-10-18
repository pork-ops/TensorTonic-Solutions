import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    A = np.array(a)
    B = np.array(b)
    dot = np.dot(a,b)
    eu_A = np.linalg.norm(A)
    eu_B = np.linalg.norm(B) 
    if not eu_A or not eu_B:
        return float(0)
    return float(dot / (eu_A*eu_B))