"""Simple replay memory for RL.

Taken from the `Pytorch documentation`__.

.. __ https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""


import random
from collections import deque

from shoggoth.proxies import Transitions


class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def add(self, transitions: Transitions):
        """Save a transition"""
        self.memory.append(transitions)

    def __next__(self):
        return random.sample(self.memory, self.batch_size)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.memory)
