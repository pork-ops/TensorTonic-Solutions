import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary.
    """
    N = np.zeros(policy_logits.shape, dtype=np.int64) # Number of times the edge i was visited
    W = np.zeros(policy_logits.shape) # total value accumulated trough the edge
    Q = np.zeros(policy_logits.shape) # is the mean value of the edge
    
    legal_actions = np.where(legal_mask, policy_logits, -np.inf)
    M = np.max(legal_actions)
    
    # prior prob assigned from the policy network. This has to sum 1 
    P = np.exp(legal_actions - M) / np.sum(np.exp(legal_actions - M))
    return {"N": N, "W": W, "Q": Q, "P": P}
    
    
    
    
