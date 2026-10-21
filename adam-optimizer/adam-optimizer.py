import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,      # m in t-1
    v: list,      # v in t-1
    t: int,       # t value
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Change to numpy arrays
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    # Raw values
    m_new = beta1 * m + (1-beta1) * grad
    v_new = beta2 * v + (1-beta2) * np.power(grad, 2)

    # Correction for low ts
    m_new_corrected = m_new/(1-np.power(beta1,t))
    v_new_corrected = v_new/(1-np.power(beta2,t))

    # Param calculation
    param_new = param - lr * (m_new_corrected/(np.sqrt(v_new_corrected) + eps))

    return (param_new, m_new, v_new)