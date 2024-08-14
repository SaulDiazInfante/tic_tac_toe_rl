import numpy as np
import random
import tic_tac_toe_rl.tic_tac_toe_rl as trl
from alive_progress import alive_bar


def q_learning_tic_tac_toe(players, par, Q):
    """
    
    Parameters
    ----------
    par
    Q
    players
    num_episodes

    Returns
    -------

    """
    num_episodes = par['num_episodes']
    exploration_rate = par['exploration_rate']
    num_players = len(players)
    with alive_bar(num_episodes, bar='blocks', spinner='twirls') as bar:
        for j in np.arange(num_episodes):
            board = np.array(
                    [
                            ['-', '-', '-'],
                            ['-', '-', '-'],
                            ['-', '-', '-']
                            ]
                    )
            current_player = random.choice(players)
            game_over = False
            while not game_over:
                action = trl.choose_action(board, exploration_rate, Q)
                
                # Make the chosen move
                row, col = action
                board[row, col] = current_player
                # Check if the game is over
                game_over, winner = trl.is_game_over(board)
                if game_over:
                    # Update the Q-table with the final reward
                    if winner == current_player:
                        reward = 1
                    elif winner == 'draw':
                        reward = 0.5
                    else:
                        reward = 0
                    Q = trl.update_q_table(
                            trl.board_to_string(board),
                            action,
                            board,
                            reward,
                            Q,
                            par
                            )
                else:
                    # Switch to the next player
                    current_player = players[
                        (players.index(current_player) + 1) % num_players
                        ]
                    # Update the Q-table with greedy actions
                    if not game_over:
                        next_state = trl.board_next_state(
                                action, board, players
                                )
                        Q = trl.update_q_table(
                                trl.board_to_string(board),
                                action,
                                next_state,
                                0,
                                Q,
                                par
                                )
                # Decay the exploration rate
            exploration_rate *= 0.99
            bar()
    par['exploration_rate'] = exploration_rate
    return Q, par


def play_against_entrained_agent(players, par, Q):
    """

    Returns
    -------
    object
    """
    board = np.array(
            [['-', '-', '-'],
             ['-', '-', '-'],
             ['-', '-', '-']]
            )
    
    current_player = random.choice(players)
    game_over = False
    num_players = len(players)
    winner = None
    # ...
    
    while not game_over:
        if current_player == 'X':
            # Human player's turn
            trl.print_board(board)
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column (0-2): "))
            action = (row, col)
        else:
            # Trained agent's turn
            exploration_rate = 0
            action = trl.choose_action(board, exploration_rate, Q)
        row, col = action
        board[row, col] = current_player
        game_over, winner = trl.is_game_over(board)
        if game_over:
            trl.print_board(board)
            if winner == 'X':
                print("Human player wins!")
            elif winner == 'O':
                print("Agent wins!")
            else:
                print("It's a draw!")
        else:
            current_player = players[
                (players.index(current_player) + 1) % num_players]
    
    return winner
