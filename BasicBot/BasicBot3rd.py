#-*- coding: utf-8 -*-

import sys
#ai_data_path = '../FTG4.30/data/aiData/BasicBot'
#ai_data_path = '../FigthingICE/data/aiData/BasicBot'
ai_data_path = '/root/userspace/BasicBot'

sys.path.append(ai_data_path)
import argparse
import importlib
import logging
import os
import re
import sys
import time
import traceback
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np
try:
    from tqdm import trange
except ImportError as exc:
    def _trange(n, *args, **kwargs):
        logger.info('Install "tqdm" to see progress bar. '
                    'Type "pip install tqdm" in terminal')
        return range(n)
    trange = _trange

from Bot import FTGEnv
from Bot import make_monitor
from Bot import resolution
from Bot import timer
from Bot import frames
from Bot import model_file
from Bot import Net
from Bot import left_actions, right_actions, energy_cost
from Bot import energy_scale
from Bot import bot_name
from Bot import log_file
from Bot import batch_size
from Bot import make_dir
from Bot import epsilon
from Bot import make_csv_logger
from Bot.Config import FTG_PATH

sys.path.append(os.path.join(FTG_PATH, 'python'))
from py4j.java_gateway import get_field
from py4j.java_gateway import set_field


logger = logging.getLogger('BasicBot')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
# logger can printout message from BasicBot(AIInterface)
# Because, Fighting Game Platform block print's functionality,
# We can't see error message, also can't see print message for debugging
# logging module is good solution for this case


class BasicBot3rd(object):
    def __init__(self, gateway):
        self.gateway = gateway
        self.frames = frames
        self.frame_count = 0

        # save short period of data to make 4 channel data
        self.capacity = resolution[0] * 5
        self.screens = list()
        self.actions = list()
        self.hps = list()
        self.energy = list()
        self.controllable = list()

        # FTGEnv object set this
        # use this to avoid change API of this bot
        #print("PPP", model_file)

        model_file = "/root/userspace/BasicBot/BasicBot3rd.pt"
        self.net = Net(resolution, len(left_actions)).cuda()
        self.net.load_state_dict(torch.load(model_file))
        #Coach().load(model_file)
        self.memorize = lambda s1, a, s2, done, r, energy: None  # do nothing

    def getCharacter(self):
        return "ZEN"

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.screenData = None

        self.player = player
        self.gameData = gameData
        return 0

    def getInformation(self, frameData):
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        # logger.info('x: {}, y: {}, z: {}'.format(x, y, z))
        pass
    	
    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def input(self):
        return self.inputKey

    def act(self, state, eps=-1):
        # this method override all BasicBot objects' act
        screens, energy = state

        if random.random() < eps:
            # exploration
            logger.debug('select random action')
            return random.randint(0, len(left_actions)-1)
        else:
            # exploitation
            logger.debug('select best action')
            screens = screens.reshape([1, resolution[0], resolution[1], resolution[2]])
            screens = Variable(torch.from_numpy(screens)).cuda()
            energy = Variable(torch.FloatTensor([[energy]])).cuda()
            q_values = self.net(screens, energy)
            #FIXME
            #print(q_values)
            #print(q_values.shape)
            #return q_values.data.max(1)[1][0][0]
            return q_values.data.argmax(1).item()

    def processing(self):
        try:
            # initialize when game start (first processing call?)
            if self.frameData.getEmptyFlag() or self.frameData.getRemainingTime() <= 0:
                self.isGameJustStarted = True
                del self.screens[:]
                del self.actions[:]
                del self.hps[:]
                del self.energy[:]
                del self.controllable[:]
                self.frame_count = self.frames
                return

            # if self.cc.getSkillFlag():
            #     self.inputKey = self.cc.getSkillKey()
            #     return

            logger.debug('{} {} {}'.format(get_field(self.inputKey, 'L'),
                                           get_field(self.inputKey, 'R'),
                                           get_field(self.inputKey, 'B')))

            # update state
            displayBuffer = self.screenData.getDisplayByteBufferAsBytes(resolution[2], resolution[1], True)
            # rescaled to [-1, 1] and change data type np.float32
            screen = np.frombuffer(displayBuffer, dtype=np.int8).reshape(resolution[1:]) / 127.0
            screen = screen.astype(np.float32)

            # opponent always inverted
            if self.player:
                screen = -screen
            self.screens.append(screen)

            my_char = self.frameData.getCharacter(self.player)
            # Bug was here
            opp_char = self.frameData.getCharacter(not self.player)
            self.hps.append((my_char.getHp(), opp_char.getHp()))
            self.energy.append(my_char.getEnergy() / energy_scale)  # energy scaled

            # set action
            action_continue = self.actions and self.actions[-1] is not None \
                              and self.frame_count < self.frames
            # self.controllable.append(get_field(my_char, 'control'))
            self.controllable.append(not self.cc.getSkillFlag())
            controllable = self.controllable[-1]

            # if get_field(my_char, 'front'):
            #     actions = left_actions
            # else:
            #     actions = right_actions

            if my_char.isFront():
                actions = right_actions
            else:
                actions = left_actions

            # if last action is set and running
            if action_continue or not controllable:
                # keep current action
                self.actions.append(self.actions[-1])
            else:
                # take new action
                self.cc.skillCancel()
                action_idx = self.act(self._get_recent_state())
                self.actions.append(action_idx)
                self.frames = len(actions[action_idx])
                logger.debug('action_idx: {}'.format(action_idx))
                self.frame_count = 0

            # set key
            if self.actions and self.actions[-1] is not None:
                current_action = actions[self.actions[-1]]
                if self.frame_count < len(current_action):
                    self.inputKey.empty()
                    action_keys = current_action[self.frame_count]
                    for key in action_keys:
                        if key in 'ABCDLRU':
                            set_field(self.inputKey, key, True)
            else:
                self.inputKey.empty()

            if getattr(self, '_on_train', False):
                # store new state action tuple and stream to monitor
                self.save_transaction(actions)
            self.frame_count += 1

        except Exception as exc:
            logger.error(traceback.format_exc())

    def save_transaction(self, actions):
        # save state and action tuple using Coach's memorize
        try:
            if len(self.screens) > self.capacity:
                del self.screens[0]
                del self.actions[0]
                del self.hps[0]
                del self.energy[0]

            if len(self.screens) == self.capacity:
                assert(len(self.screens) == len(self.actions)
                       == len(self.hps) == len(self.energy))
                s1 = np.stack(
                    [self.screens[0],  # 1st frame
                     self.screens[4],  # 2nd frame
                     self.screens[8],  # 3rd frame
                     self.screens[12]])  # 4th frame
                s2 = np.stack(
                    [self.screens[4],  # 2nd frame
                     self.screens[8],  # 3rd frame
                     self.screens[12],  # 4th frame
                     self.screens[16]])  # 5th frame
                a = self.actions[0]
                # reward calculation
                my_hp_1, opp_hp_1 = self.hps[0]
                my_hp_2, opp_hp_2 = self.hps[4]
                energy = self.energy[0]
                r = (opp_hp_1 - opp_hp_2) - (my_hp_1 - my_hp_2) if energy_cost[a] > energy else -1.0

                if self.controllable[0]:
                    # memorize sample when character is controllable
                    # do not use game end in this platform
                    # done is always false
                    self.memorize(s1, a, s2, False, r, energy)

                # for calculate score
                #Coach().add_reward(r)

                # if there are debug port (process queue for monitoring code)
                # send current state and action  through this
                if hasattr(self, 'debug_port'):
                    cnt = getattr(self, '_debug_port_cnt', 0)
                    if cnt == 0:
                        screens, energy = self._get_recent_state()
                        info = dict(action=str(actions[self.actions[-1]]),
                                    energy=int(energy * energy_scale),
                                    reward=r)
                        self.debug_port.put((screens, info))
                    setattr(self, '_debug_port_cnt', (cnt + 1) % 4)

        except Exception as exc:
            logger.error(traceback.format_exc())

    def _get_recent_state(self):
        if len(self.screens) == self.capacity:
            screen = np.stack([self.screens[-13],  # 1st frame
                               self.screens[-9],  # 2nd frame
                               self.screens[-5],  # 3rd frame
                               self.screens[-1]])  # 4st frame
        else:
            screen = np.stack([self.screens[-1],
                               self.screens[-1],
                               self.screens[-1],
                               self.screens[-1]])
        energy = self.energy[-1]
        return screen, energy

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]

