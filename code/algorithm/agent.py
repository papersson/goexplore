import numpy as np


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

    def __repr__(self):
        return 'RandomAgent'


class ActionRepetitionAgent:

    # From https://github.com/uber-research/go-explore/blob/240056852514ffc39f62d32ae7590a39fd1814b9/policy_based/goexplore_py/explorers.py#L26
    # Repeats actions with 95% probability
    def __init__(self, action_space, mean_repeat=20):
        self.action_space = action_space
        self.mean_repeat = mean_repeat
        self.action = 0  # noop
        self.remaining = 0

    def act(self):
        if self.remaining <= 0:
            self.remaining = np.random.geometric(1 / self.mean_repeat)
            # self.action = random.choice(self.action_space)
            self.action = self.action_space.sample()
        self.remaining -= 1
        return self.action

    def __repr__(self):
        return 'ActionRepetitionAgent'
