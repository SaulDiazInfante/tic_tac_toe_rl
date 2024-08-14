import random
import numpy as np
import tic_tac_toe_rl.tic_tac_toe_rl as trl
import tic_tac_toe_rl.ql_tic_tac_toe as tql


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

# print('a_t[i, j] = [{0}, {1}]'.format(action[0], action[1]))

Q, par = tql.q_learning_tic_tac_toe(players, par, Q)
winner = tql.play_against_entrained_agent(players, par, Q)