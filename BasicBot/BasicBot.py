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
from Bot import Coach
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


class BasicBot(object):
    def __init__(self, gateway):
        self.gateway = gateway
        self.frames = frames
        self.frame_count = 0
        self.old_hp = None

        # save short period of data to make 4 channel data
        self.capacity = resolution[0] * 5
        self.screens = list()
        self.actions = list()
        self.hps = list()
        self.energy = list()
        self.controllable = list()

        # FTGEnv object set this
        # use this to avoid change API of this bot
        on_train = getattr(self, '_on_train', False)

        # when start game, subscribe self to Coach object
        # and override act, and memorise
        if on_train:
            Coach().load(model_file)
            Coach().trainees.append(self)
            self.act = functools.partial(Coach().act, eps=1.0)
            self.memorize = Coach().memorize
        else:
            Coach().load(model_file)
            self.act = functools.partial(Coach().act, eps=-1)  # always do best action, epsilon == -1
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
                #my_hp_2, opp_hp_2 = self.hps[4]
                if self.old_hp is not None and (self.hps[4][0] >= self.old_hp[0] and self.hps[4][1] >= self.old_hp[1] ):
                    my_hp_2, opp_hp_2 = self.old_hp
                else:
                    my_hp_2, opp_hp_2 = self.hps[4]
                self.old_hp = (my_hp_1, opp_hp_1)
                energy = self.energy[0]
                r = (opp_hp_1 - opp_hp_2) - (my_hp_1 - my_hp_2) if energy_cost[a] > energy else 0#-1.0

                if self.controllable[0]:
                    # memorize sample when character is controllable
                    # do not use game end in this platform
                    # done is always false
                    self.memorize(s1, a, s2, False, r, energy)

                # for calculate score
                Coach().add_reward(r)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="BasicBot example for Visual based Fighting Game AI Platform")
    parser.add_argument('train_or_test', nargs='+', choices=['train', 'test'],
                        help='Select Traning mode or Testing mode')
    parser.add_argument('--max-epoch', default=None, type=int, help='Max epochs for training')
    parser.add_argument('-ts', '--training-step', type=int, default=10000, help='set how many frame use for training')
    parser.add_argument('-es', '--evaluation-sec', type=int, default=60,
                        help='set how many time(seconds) use for evaluation (testing) without learning')
    parser.add_argument('-o', '--opponent', default='SandBag',
                        help='''opponent name
  - Python bot: opponent class name and filename shoud be same and it placed in same directory
  - Java bot: should be add character type like 'MctsAi:LUD', however it is not test enough''')
    parser.add_argument('-n', '--n-cpu', default=3, type=int,
                        help='set how many games launch, it should lower than number of CPU cores, '
                             'also considering limits of other computing resources')
    parser.add_argument('-p', '--path', default='.',
                        help='create this directory if it is not exists '
                             'and write csv file as a log and store networks parameters'
                             '(not working on python 2')
    parser.add_argument('--port', default=6000, type=int, help='help')
    parser.add_argument('-r', '--render', default='single', choices=['none', 'single', 'all'], help='game server port')
    parser.add_argument('-v, --verbose', dest='verbose', action='store_true', help='show debug level message')
    parser.set_defaults(verbose=False)
    parser.add_argument('-m', '--monitor', default='none',
                        choices=['none', 'pygame', 'matplotlib'],
                        help='show current input screen data and reward value and out action, '
                             'two types supported pygame and matplotlib')
    parser.add_argument('--starts-with-energy',dest='starts_with_energy', action='store_true',
                        help='get max energy for each character when start a game')
    parser.set_defaults(starts_with_energy=False)
    parser.add_argument('--disable-tqdm', dest='use_tqdm', action='store_false', help='disable tqdm')
    parser.set_defaults(use_tqdm=True)
    parser.add_argument('--n-games', type=int, default=1000000)
    args = parser.parse_args()

    logger.info('Launch {:d} environment(s)'.format(args.n_cpu))
    logger.info('{} vs. {}'.format(bot_name, args.opponent))

    if not args.use_tqdm:
        logger.info('Disable tqdm.trange')

        def _trange(*args, **kwargs):
            return range(*args)
        # override tqdm trange
        globals()['trange'] = _trange

    opponent = args.opponent
    if re.search("^[a-z0-9]+:(ZEN|LUD|GARNET)$", args.opponent, re.I):
        # for Java bot: eg. 'MctsAi:LUD'
        logger.info("Set java opponent: {}".format(args.opponent))
    else:
        # for python bot: bot's class name and file name should same,
        # the file should be place where this script can import
        try:
            mod = importlib.import_module(args.opponent)
            opponent = getattr(mod, args.opponent)
            logger.info("Set python opponent: {}".format(args.opponent))
        except ImportError:
            logger.error("Can't import {}".format(args.opponent))
            sys.exit(-1)

    root_path = args.path
    logger.info('Set path: {}'.format(root_path))
    make_dir(root_path)
    step_per_training = args.training_step
    seconds_per_testing = args.evaluation_sec

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Verbose mode on')

    logger.info('** Start {}ING ** '.format(args.train_or_test[0].upper()))
    # Train start
    if 'train' in args.train_or_test:
        logger.debug('Open {} as logfile to write'.format(os.path.join(root_path, log_file)))
        write_to_csv = make_csv_logger(root_path, log_file)

    ai_monitor = make_monitor(args.monitor, env_id=0, player_no=1)

    envs = list()
    logger.info('Args: {}'.format(args))
    try:
        if 'train' in args.train_or_test:
            for n in range(args.n_cpu):
                disable_window = not (args.render == 'on' or args.render == 'single' and n == 0)
                env = FTGEnv(n, BasicBot, opponent, port=args.port + n,
                            inverted_player=1,
                            disable_window=disable_window,
                            starts_with_energy=args.starts_with_energy,
                            train='train' in args.train_or_test, verbose=False)
                env.run(block=False, ai_monitor=ai_monitor, n_game=args.n_games)
                envs.append(env)

            while Coach().memory.size < batch_size:
                time.sleep(0.016 * frames)
                pass

            epoch = 0
            start_time = timer()
            highest_score = -1e-10

            while True:
                Coach().training(epsilon)
                q_values = np.zeros(step_per_training)
                losses = np.zeros(step_per_training)
                for step in trange(step_per_training, desc='Training {}'.format(epoch)):
                    max_q, loss = Coach().learn()
                    q_values[step] = max_q.mean()
                    losses[step] = loss
                training_score = Coach().get_rewards_stat()

                Coach().testing()
                for step in trange(seconds_per_testing, desc='Testing {}'.format(epoch)):
                    time.sleep(1)
                testing_score = Coach().get_rewards_stat()

                logger.info('Epoch: {} Testing: {:.3f} ± {:.3f} Training: {:.3f} ± {:.3f} Loss: {:.9f} Q: {:.3f} ε: {:.5f}'.format(
                    epoch, testing_score.mean(), testing_score.std(), training_score.mean(), training_score.std(),
                    losses.mean(), q_values.mean(), epsilon))

                write_to_csv(epoch, start_time, testing_score, training_score, losses)

                if testing_score.mean() > highest_score:
                    highest_score = testing_score.mean()
                    Coach().save(model_file, hint=highest_score)
                    logger.info('New highest score: {}'.format(highest_score))

                # reduce epsilon to minimum (0.1) until half of the max_epoch
                epsilon_delta = 0.05 if args.max_epoch is None else 1.0 / args.max_epoch * 2
                epsilon = max(0.1, epsilon - epsilon_delta)
                epoch += 1
                if args.max_epoch is not None and epoch > args.max_epoch:
                    break

        if 'test' in args.train_or_test:
            disable_window = not (args.render == 'on' or args.render == 'single')
            # env = FTGEnv(0, BasicBot, opponent, port=args.port,
            # sys.path.append(os.path.join(FTG_PATH, 'python'))
            env = FTGEnv(0, BasicBot, opponent, port=args.port,
                        inverted_player=2,
                        disable_window=disable_window,
                        starts_with_energy=args.starts_with_energy,
                        train='train' in args.train_or_test, verbose=args.verbose)
            env.run(block=True, ai_monitor=ai_monitor, n_game=args.n_games)
            envs.append(env)
    except KeyboardInterrupt:
        if len(envs) > 0:
            env = envs[0]
        env.gateway.shutdown_callback_server(raise_exception=False)
        env.gateway.shutdown(raise_exception=False)
        sys.stdout.flush()
        
    logger.info('** Stop {}ING ** '.format(args.train_or_test[0].upper()))
    logger.info('Press ctrl + c many times to exit')
