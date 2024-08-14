import numpy as np
import random


def print_board(board):
    """

    Parameters
    ----------
    board

    Returns
    -------
    object
    """
    for row in board:
        print('  |  '.join(row))
        print('---------------')


def board_to_string(board):
    """ Cast the board to a flat string
    Parameters
    ----------
    board

    Returns
    
    str_board
        string representation of board
    -------
    Examples
    --------
    >>> board = np.array(
            [['-', '-', '-'],
             ['-', '-', '-'],
             ['-', '-', '-']]
            )
    >>> board_to_string(board)
    >>> print (board)
    '---------'
    """
    str_board = ''.join(board.flatten())
    return str_board


def is_game_over(board):
    """
    
    :param board:
    :return:
    """
    # Check rows for winning condition
    for row in board:
        if len(set(row)) == 1 and row[0] != '-':
            return True, row[0]
    # Check columns
    for col in board.T:
        if len(set(col)) == 1 and col[0] != '-':
            return True, col[0]
        # diagonals
    if len(set(board.diagonal())) == 1 and board[0, 0] != '-':
        return True, board[0, 0]
    if len(set(np.fliplr(board).diagonal())) == 1 and board[0, 2] != '-':
        return True, board[0, 2]
    if '-' not in board:
        return True, 'draw'
    return False, None


def choose_action(board, exploration_rate, Q):
    """
    :param Q:
    :param board:
    :param board:
    :param exploration_rate:
    :return:
    """
    state = board_to_string(board)
    # Exploration-exploitation trade-off
    if random.uniform(0, 1) < exploration_rate or state not in Q:
        # Choose a random action
        empty_cells = np.argwhere(board == '-')
        action = tuple(random.choice(empty_cells))
    else:
        # Choose the action with the highest Q-value
        q_values = Q[state]
        empty_cells = np.argwhere(
                board == '-'
                )  # returns indices of the empty cells in the board.
        empty_q_values = [
                q_values[cell[0],
                cell[1]] for cell in empty_cells
                ]  # retrieves Q-values
        # corresponding to each empty cells.
        max_q_value = max(
                empty_q_values
                )  # find the maximum Q-value among the empty cells Qvalue
        max_q_indices = [
                i for i in range(len(empty_cells))
                if empty_q_values[i] == max_q_value
                ]
        max_q_index = random.choice(max_q_indices)
        action = tuple(
                empty_cells[max_q_index]
                )
    return action


def board_next_state(cell, board, players):
    """

    Parameters
    ----------
    cell
    board
    players
    """

    next_state = board.copy()  # create a copy of current board state
    next_state[cell[0], cell[1]] = players[0]
    return next_state


def update_q_table(state, action, next_state, reward, Q, par):
    """

    Parameters
    ----------
    state
    action
    next_state
    reward
    Q
    par
    """
    learning_rate = par['learning_rate']
    discount_factor = par['discount_factor']
    
    q_values = Q.get(state, np.zeros((3, 3)))
    # Calculate the maximum Q-value for the next state
    next_q_values = Q.get(board_to_string(next_state), np.zeros((3, 3)))
    max_next_q_value = np.max(next_q_values)
    
    # Q-learning update equation
    q_values[action[0], action[1]] += (
        learning_rate * (
            reward + discount_factor * max_next_q_value
            - q_values[action[0], action[1]]
        )
    )
    Q[state] = q_values
    return Q
