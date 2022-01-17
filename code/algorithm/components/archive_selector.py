import numpy as np


class ReverseCountSelector:
    def sample(self, archive):
        visits = [cell.visits for cell in archive.values()]
        # Example
        # [1, 4, 5, 4] ->
        # [5, 2, 1, 2]
        weights = [max(visits) + 1 - v for v in visits]
        probs = [w / sum(weights) for w in weights]
        return np.random.choice(list(archive.values()), 1, p=probs)[0]


class UberSelector:
    # def sample(self, archive):
    #     visits = [cell.visits for cell in archive.values()]
    #     weights = [1 / np.log(c.visits + 1) for c in archive.values()]
    #     probs = [w / sum(weights) for w in weights]
    #     return np.random.choice(list(archive.values()), 1, p=probs)[0]
    def __init__(self):
        self.cells = []

    def update(self, cells):
        self.cells = cells
        weights = np.array([c.get_weight() for c in cells])
        self.probs = weights / sum(weights)

    def sample(self):
        # visits = [cell.visits for cell in archive.values()]
        # weights = [1 / np.log(c.visits + 1) for c in archive.values()]
        # probs = [w / sum(weights) for w in weights]
        return np.random.choice(self.cells, 1, p=self.probs)[0]
