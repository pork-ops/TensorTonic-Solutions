import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    # Should have used sets to prevent duplicates current code is super inefficient using lists...
    board = np.array(board)
    color = board[row][col]
    if color == 0:
        raise Exception("You need to pick a stone!")

    pending_to_process = []
    for n in [[row, col - 1], [row, col + 1], [row - 1, col], [row + 1, col]]:
        if n[0] < board.shape[0] and n[1] < board.shape[1] and n[0] >= 0 and n[1] >= 0:
            pending_to_process.append(n)
        
    visited = []
    group = [[row, col]]
    liberties = []
    
    while pending_to_process:
        current = pending_to_process.pop()
        visited.append(current)
        
        if board[current[0]][current[1]] == color:
            if current not in group:
                group.append([current[0],current[1]])
                row = current[0]
                col = current[1]
                new_pending = [row, col - 1], [row, col + 1], [row - 1, col], [row + 1, col]
                for n in new_pending: 
                    if n not in visited:
                        if n[0] < board.shape[0] and n[1] < board.shape[1] and n[0] >= 0 and n[1] >= 0:
                            pending_to_process.append(n)
        elif board[current[0]][current[1]] == 0 and current not in liberties:
            liberties.append(current)
    return sorted(group), sorted(liberties)