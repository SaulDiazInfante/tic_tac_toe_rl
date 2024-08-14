import random
import numpy as np
from tic_tac_toe_rl import tic_tac_toe_rl as trl

board = np.array(
        [['-', '-', '-'],
         ['-', '-', '-'],
         ['-', '-', '-']]
        )
players = ['X', 'O']
num_players = len(players)
Q = {}
learning_rate = 0.001
discount_factor = 0.9
exploration_rate = 0.5
num_episodes = 10000

par = {
    'learning_rate': learning_rate,
    'discount_factor': discount_factor,
    'exploration_rate': exploration_rate,
    'num_episodes': num_episodes
}

trl.print_board(board)
trl.board_to_string(board)
empty_cells = np.argwhere(board == '-')
action = tuple(random.choice(empty_cells))
print(action)
agent_wins = 0

# Main Q-learning algorithm
for episode in range(num_episodes):
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
        # Choose an action based on the current state
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
        
        # Update the Q-table based on the immediate reward and the next state
        if not game_over:
            next_state = trl.board_next_state(action, board, players)
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

# Play against the trained agent
board = np.array(
        [['-', '-', '-'],
         ['-', '-', '-'],
         ['-', '-', '-']]
        )

current_player = random.choice(players)
game_over = False

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
        action = trl.choose_action(board, 0, Q)
    
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

# agent_win_percentage = (agent_wins / num_games) * 100
# print("Agent win percentage: {:.2f}%".format(agent_win_percentage))
# Main Q-learning algorithm
num_draws = 0  # Counter for the number of draws
agent_wins = 0  # Counter for the number of wins by the agent

for episode in range(num_episodes):
    board = np.array(
            [['-', '-', '-'],
             ['-', '-', '-'],
             ['-', '-', '-']]
            )
    
    current_player = random.choice(
        players
        )  # Randomly choose the current player
    game_over = False
    
    while not game_over:
        action = trl.choose_action(board, exploration_rate, Q)  # Choose an
        # action using the exploration rate
        
        row, col = action
        board[
            row, col] = current_player  # Update the board with the current
        # player's move
        
        game_over, winner = trl.is_game_over(board)  # Check if the game is
        # over and determine the winner
        
        if game_over:
            if winner == current_player:  # Agent wins
                reward = 1
                agent_wins += 1
            elif winner == 'draw':  # Game ends in a draw
                reward = 0
                num_draws += 1
            else:  # Agent loses
                reward = -1
            trl.update_q_table(
                trl.board_to_string(board),
                action,
                board,
                reward,
                Q,
                par
            )
            # Update the Q-table
        else:
            current_player = players[
                (players.index(current_player) + 1) % num_players
                ]  # Switch to the next player
        if not game_over:
            next_state = trl.board_next_state(action, board, players)
            trl.update_q_table(
                trl.board_to_string(board),
                action,
                next_state,
                0,
                Q,
                par
            )  # Update the Q-table with the next state
    exploration_rate *= 0.99  # Decrease the exploration rate over time

# Play multiple games between the trained agent and itself
# agent_win_percentage = (agent_wins / num_games) * 100
# draw_percentage = (num_draws / num_games) * 100

# print("Agent win percentage: {:.2f}%".format(agent_win_percentage))
# print("Draw percentage: {:.2f}%".format(draw_percentage))
