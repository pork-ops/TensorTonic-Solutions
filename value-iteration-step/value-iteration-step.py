import math 

def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    # Write code here
    # Problem request to not use np
    result = [0 for _ in values]
    
    for i in range(len(values)):                   # states
        max_value = -math.inf
        for j in range(len(transitions[0])):
            actions = transitions[i][j]
            sum_over_future = sum(probability * value for probability, value in zip(actions, values))
            new_value = rewards[i][j] + gamma * sum_over_future
            if new_value > max_value:
                result[i] = new_value
                max_value = new_value
    return result