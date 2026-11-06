def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    # Write code here
    td_errors = []
    for i in range( len(rewards) - 1, -1, -1):
        td_errors.append(rewards[i] + gamma * values[i+1] - values[i])

    print(td_errors)

    advantages = [td_errors[0]]
    for i in range(1, len(td_errors)):
        advantages.append(td_errors[i] + gamma * lam * advantages[-1])

    print(advantages)
    return advantages[::-1]