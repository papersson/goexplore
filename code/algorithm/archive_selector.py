import numpy as np


class ReverseCountSelector:
    def sample(self, archive):
        visits = [cell.visits for cell in archive.values()]
        weights = [max(visits) + 1 - v for v in visits]
        probs = [w / sum(weights) for w in weights]
        return np.random.choice(list(archive.values()), 1, p=probs)[0]


class UberSelector:
    def sample(self, archive):
        visits = [cell.visits for cell in archive.values()]
        weights = [1 / np.log(c.visits + 1) for c in archive.values()]
        probs = [w / sum(weights) for w in weights]
        return np.random.choice(list(archive.values()), 1, p=probs)[0]
