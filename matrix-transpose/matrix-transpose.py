import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    # A is assumed to be a two-dimensional Python list of shape (N, M)
    # return np.array(A).transpose() # easy way
    try:
        n = len(A)
        m = len(A[0])
    except:
        print("The matrix has to bee two-dimensional")
    A_T = np.empty([m, n])
    for i in range(n):
        for j in range(m):
            A_T[j][i] = A[i][j]
    return A_T
    
