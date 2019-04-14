# -*-coding: utf-8 -*-

import abc
import logging
from six import with_metaclass
from threading import Lock
import random
import os
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Bot import ReplayMemory
from Bot import resolution
from Bot import left_actions
from Bot import energy_scale, distance_scale
from Bot import energy_cost
from Bot import learning_rate
from Bot import memory_size
from Bot import make_dir
from Bot import root_path
from Bot import bot_name
from Bot import FTG_PATH
from Bot import batch_size
from Bot import gamma
from Bot import reward_scale
import traceback


logger = logging.getLogger('BasicBot')


class Net(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(Net, self).__init__()
        channel, height, width = input_shape
        # [-1, 4, 38, 57]
        self.conv1 = nn.Conv2d(channel, 8, 6, stride=3)
        # [-1, 8, 11, 18]
        self.conv2 = nn.Conv2d(8, 8, 3, stride=2)
        # [-1, 8, 5, 8]
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8 * 5 * 8 + 2, 128)
        self.fc2 = nn.Linear(128, n_outputs)

    def forward(self, s1, eng, distance):
        try:
            s1_org = s1
            s1 = F.relu(self.conv1(s1_org))
            s1 = F.relu(self.conv2_drop(self.conv2(s1)))
            s1 = s1.view(-1, 8 * 5 * 8)
            x = torch.cat((s1, eng, distance), 1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x
        except Exception as exc:
            logger.error(exc)
            exit(-1)


class Singleton(abc.ABCMeta):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]


LearnerLock = Lock()
#net = Net(resolution, len(left_actions)).cuda()
net = Net(resolution, len(left_actions)).cuda()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.05)
loss_func = torch.nn.MSELoss()


class Coach(with_metaclass(Singleton)):
    # - Do all jobs using neural network (learning, decision making, save and load parameters)
    # - Singleton: every objects can access this object, and others through this object (act, memorize)
    # - change training, testing mode
    def __init__(self):
        with LearnerLock:
            self.lock = LearnerLock
            # every BasicBot objects
            self.trainees = list()
            self.reward_log = [0]
            self.memory = ReplayMemory(resolution, memory_size)

    def save(self, model_file='BasicBot.pt', **kwargs):
        # save model's parameters
        # there are two files to save
        # - model.pt: saved in current working directory or root_path, current highest performance,
        #   save method call when new high performance archived
        # - model.pt-{id}: to preserve past best parameters, if pass over mean score (kwargs[hint]) from caller
        #   {id} == current score
        with self.lock:
            # save "./model.pt"
            model_path = os.path.join(root_path, model_file)
            torch.save(net.state_dict(), model_path)
            logger.info('Model file({}) saved'.format(model_path))

            # save backup "./models/model.pt-{id}"
            make_dir(os.path.join(root_path, 'models'))
            # {id}: current score (the highest score or current time
            model_path_aux = os.path.join(
                root_path, 'models', '{}-{}'.format(
                    model_file, kwargs.get('hint', time.ctime())))
            torch.save(net.state_dict(), model_path_aux)
            logger.info('Model file aux({}) saved'.format(model_path_aux))

    def load(self, model_file=None):
        # traceback.print_stack() 
        if model_file is None:
            model_file = '{}.pt'.format(__file__[:-3])

        paths = [os.path.join(FTG_PATH, 'data/aiData', bot_name, model_file),
                 os.path.join(root_path, model_file)]

        model_path = ""
        for path in paths:
            if os.path.exists(path):
                model_path = path
                break

        with self.lock:
            if os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path))
                logger.info('Model file({}) loaded'.format(model_path))
            else:
                logger.warning('Model file dose not exixts, '
                               'please place model file (BasicBot.pt) in current working directory '
                               'or {}/data/aiData/BasicBot'.format(FTG_PATH))

    def act(self, state, eps):
        # this method override all BasicBot objects' act
        with self.lock:  # every BasicBot objects can call this
            screens, energy, distance = state

            if random.random() < eps:
                # exploration
                logger.debug('select random action')
                sample = random.randint(0, len(left_actions)-1)
#                return sample
                while True:
                    if energy_cost[sample] / energy_scale <= energy:
                        return sample
                    sample = random.randint(0, len(left_actions)-1)
            else:
                # exploitation
                logger.debug('select best action')
                screens = screens.reshape([1, resolution[0], resolution[1], resolution[2]])
                screens = Variable(torch.from_numpy(screens)).cuda()
                energy = Variable(torch.FloatTensor([[energy]])).cuda()
                distance = Variable(torch.FloatTensor([[distance]])).cuda()
                q_values = net(screens, energy, distance)
                #FIXME
                #print(q_values)
                #print(q_values.shape)
                #return q_values.data.max(1)[1][0][0]
                sample = q_values.data.argmax(1).item()
#                return sample
                for i in range(200):
                    if energy_cost[sample] / energy_scale <= energy:
                        return sample
                    q_values.data[0][sample] = -100
                    sample = q_values.data.argmax(1).item()
                #return q_values.data.argmax(1).item()

    def memorize(self, s1, a, s2, done, r, energy, distance):
        # this method override all BasicBot objects' memorize
        self.memory.add(s1, a, s2, done, r, energy, distance)

    def learn(self):
        s1, a, s2, _, r, energy, distance = self.memory.sample(batch_size)
        s1 = Variable(torch.from_numpy(s1)).cuda()
        s2 = Variable(torch.from_numpy(s2)).cuda()
        r /= reward_scale
        energy = energy.reshape((batch_size, 1))
        energy = Variable(torch.from_numpy(energy)).cuda()
        distance = distance.reshape((batch_size, 1))
        distance = Variable(torch.from_numpy(distance)).cuda()

        with self.lock:  # get lock when only change parameters
            q1 = net(s1, energy, distance).data.cpu().numpy()
            max_q = np.max(q1, axis=1)
            max_q2 = net(s2, energy, distance).data.max(1)[0].cpu().numpy().reshape(batch_size)

            q1[np.arange(batch_size), a] = r + gamma * max_q2

            optimizer.zero_grad()
            target = Variable(torch.from_numpy(q1).cuda())
            output = net(s1, energy, distance)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            logger.debug('learn one batch')
            #return max_q, loss.data[0]
            return max_q, loss.data.item()

    def training(self, eps):
        # set training mode and broadcast to all BasicBot objects
        logger.debug('Set training mode')
        for trainee in self.trainees:
            trainee.act = functools.partial(Coach().act, eps=eps)
            trainee.memorize = Coach().memorize

    def testing(self):
        # set testing mode and broadcast to all BasicBot objects
        logger.debug('Set testing mode')
        for trainee in self.trainees:
            trainee.act = functools.partial(Coach().act, eps=-1)  # -1, 0.1
            trainee.memorize = Coach().memorize

    def add_reward(self, reward):
        with self.lock:
            self.reward_log.append(reward)  # save rewards for calculate score for certain period

    def get_rewards_stat(self):
        # this method called when training or testing end
        # to calculate training score or testing score
        with self.lock:
            rewards = np.array(self.reward_log)
            del self.reward_log[:]
            self.reward_log.append(0)
            return rewards
