import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Write code here
    threshold = 1e-12
    A = np.array(A)
    n = A.shape[0]
    
    both = np.hstack((A, np.eye(n)))
    for c in range(n):
        # 1. Find row with largest value for column c 
        column = np.abs(both[c:,c])
        max_val = np.max(column)
        if max_val < threshold:
            return None
        row = np.argmax(column) + c

        # 2. Change Swap rows c and row of both
        both_aux = both[row,:].copy()
        both[row,:] = both[c,:]
        both[c,:] = both_aux

        # 3. Normalize row
        both[c,:] = both[c,:] / both[c,c]

        # 4. Elimination
        for r in range(n):
            f = both[r, c]
            if r == c: 
                continue
            both[r, :] = both[r, :] - (f * both[c, :])

    return both[:,n:]
        