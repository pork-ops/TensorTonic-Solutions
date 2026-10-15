import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def binary_cross_entropy_loss(X: np.ndarray, y: np.ndarray, p: np.ndarray):
    return np.sum(np.where(y==1, -1*np.log(p), -1*(np.log(1-p))))/X.shape[0]

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    num_features = X.shape[1]
    w = np.zeros(num_features)  # weights
    b = 0.0                     # bias
    print("Features:", num_features, "X:", X.shape, "w:", w.shape)
    for step in range(steps):
        z = np.matmul(X, w) + b
        # print(z.shape)
        p = _sigmoid(z)
        L = binary_cross_entropy_loss(X, y, p)
        # print(f"For the step {step} the loss is {L}")
        # let's calculate the gradients
        gradient_L_w = (np.matmul(np.transpose(X),(p - y))) / X.shape[0]
        gradient_L_b = np.sum(p - y) / X.shape[0]
        w = w - (lr * gradient_L_w)
        b = b - (lr * gradient_L_b)
        # print("Features:", num_features, "X:", X.shape, "w:", w.shape, "gradient_L_w:", gradient_L_w)
    return w, b