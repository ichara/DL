# -*- coding: utf-8 -*-

from random import sample
from multiprocessing import Lock
import numpy as np


class ReplayMemory:
    def __init__(self, shape, capacity):
        self.lock = Lock()  # for multiprocessing, threading
        state_shape = (capacity, shape[0], shape[1], shape[2])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)  # do not use this platform
        self.energy = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add(self, s1, action, s2, done, reward, energy):
        with self.lock:
            self.s1[self.pos, :, :, :] = s1
            self.a[self.pos] = action
            if not done:
                self.s2[self.pos, :, :, :] = s2
            self.done[self.pos] = done
            self.r[self.pos] = reward
            self.energy[self.pos] = energy

            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        with self.lock:
            return self.s1[i], self.a[i], self.s2[i], self.done[i], self.r[i], self.energy[i]
