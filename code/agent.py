import random

import numpy as np


class RandomAgent:
    def __init__(self, action_space=[0, 2, 3]):
        self.action_space = action_space

    def act(self):
        return random.choice(self.action_space)

# From https://github.com/uber-research/go-explore/blob/240056852514ffc39f62d32ae7590a39fd1814b9/policy_based/goexplore_py/explorers.py#L26
# Repeats actions with 95% probability
# TODO: is it equivalent to sticky actions?


class ActionRepetitionAgent:
    def __init__(self, action_space=[0, 2, 3], mean_repeat=20):
        self.action_space = action_space
        self.mean_repeat = mean_repeat
        self.action = 0  # noop
        self.remaining = 0

    def act(self):
        if self.remaining <= 0:
            self.remaining = np.random.geometric(1 / self.mean_repeat)
            self.action = random.choice(self.action_space)
        self.remaining -= 1
        return self.action
