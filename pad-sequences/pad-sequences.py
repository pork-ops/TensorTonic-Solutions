import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.empty((0,0), dtype=int)
    if max_len is None:
        max_len = max([len(x) for x in seqs])
    out = np.full([len(seqs), max_len],pad_value, dtype=int)
    for i in range(len(seqs)):
        current = len(seqs[i]) if len(seqs[i]) < max_len else max_len
        out[i][:current] = seqs[i][:current]
    return out