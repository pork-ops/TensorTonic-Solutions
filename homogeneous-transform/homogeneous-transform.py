import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T = np.array(T)
    points = np.atleast_2d(points)
    dim_to_return = points.shape[0]
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points.T
    result = np.dot(T, points)
    if dim_to_return == 1:
        return result[:][:3].reshape(3)
    else:
        result = result[:][:3].T
        return result