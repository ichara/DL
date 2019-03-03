# -*- coding: utf-8 -*-

import sys
import os
import time


if sys.version_info[0] == 3:
    _string = str
    make_dir = lambda path: os.makedirs(path, exist_ok=True)
    timer = time.perf_counter
elif sys.version_info[0] == 2:
    import string as _string
    make_dir = lambda path: path  # do nothing
    timer = time.clock


def _left2right(left_action):
    d = _string.maketrans('LR', 'RL')
    right_action = list()
    for a in left_action:
        if type(a) == int:
            right_action.append(a)
        else:
            right_action.append(_string.translate(a, d))
    return right_action


#FTG_PATH = '../FTG4.30'
#FTG_PATH = '../public/FightingICE'
FTG_PATH = '../FightingICE'
resolution = (4, 38, 57)
# (channel, height, width),
# channel: frames[t-3, t-2, t-1, t-0]
# FIXME AWS G2 memory error
# memory_size = 500000
memory_size = 50000
batch_size = 32
learning_rate = 0.01
gamma = 0.9999  # discount factor
#epsilon = 1.0  # initial epsilon (exploration rate) for epsilon greedy algorithm
epsilon = 0.1  # initial epsilon (exploration rate) for epsilon greedy algorithm
reward_scale = 30.0   # scaled_reward = reward / reward_scale
#reward_scale = 100.0   # scaled_reward = reward / reward_scale
energy_scale = 300.0  # scaled_energy = energy / energy_scale
frames = 4

bot_name = 'BasicBot'
log_file = '{}.csv'.format(bot_name)
model_file = '{}.pt'.format(bot_name)
root_path = '.'


# actions that BasicBot can perform
# action_seq = (
#     # ('DASH', 'DASH'),
#     # ('BACK_STEP', 'BACK_STEP'),
#     # ('2_B', '2_B'),
#
#     ('DASH',),
#     ('BACK_STEP',),
#     ('2_B',),
# )


# actions = [
#     (0, 'L', '', 'L', ''),
#     (0, 'R', '', 'R', ''),

#     (0, 'B', '', 'B', ''),  # B
#     (0, 'DB', '', 'DB', ''),  # 2_B
#     # ('DB', ' ', 'DB', ' ', 'DB', ' ', 'DB', ' '),  # 2_B

#     (2, 'D', 'DR', 'RA', ''),  # 2 3 6_A
#     (30, 'D', 'DR', 'RB', ''),  # 2 3 6_B
#     (150, 'D', 'DR', 'RC', ''),  # 2 3 6_C

#     ## ('R', 'D', 'RDA', ''),   # 6 2 3_A
#     ## ('R', 'D', 'RDB', ''),   # 6 2 3_B

#     (0, 'D', 'LD', 'LA', ''),   # 2 1 4_A
#     (50, 'D', 'LD', 'LB', ''),   # 2 1 4_B
# ]


actions = [
    #(300, 'LU', 'B', 'B', 'B'),  # FOR_JUMP _B B B
    (0, 'LU', 'B', 'B', 'B'),  # FOR_JUMP _B B B
    (0, 'LU'),  # FOR_JUMP
    (100, 'D', 'DL', 'LC'),  # STAND_D_DF_FC, 2 3 6 _ C
    (50, 'D', 'DR', 'RB'),  # STAND_D_DB_BB, 2 1 4 _ B
    (0, 'L', 'LD', 'DL', 'LA'),  # STAND_F_D_DFA, 6 2 3 _ A
    (0, 'L', 'L', 'L', ''),  # 6 6 6
    (0, 'B'),  # B
    (5, 'U', 'UD', 'DB'),   # AIR_DB
    (50, 'D', 'DR', 'RD', 'DB'),   # 2 1 4_B
]

energy_cost = [action[0] for action in actions]
left_actions = [action[1:] for action in actions]
right_actions = [_left2right(action) for action in left_actions]
