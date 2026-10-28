def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    top_k_relevant = len([reco for reco in recommended[:k] if reco in relevant ])
    precision = top_k_relevant / k
    recall = top_k_relevant / len(relevant)
    return [precision,recall]