import numpy as np
from utils.dynamicarray import DynamicArray


SIZE = 50000
def to_weight(n_visits): return 1 / np.sqrt(1 / n_visits + 1)


# class ReverseCountSelector:
#     def sample(self, archive):
#         n_visits = [cell.n_visits for cell in archive.values()]
#         # Example
#         # [1, 4, 5, 4] ->
#         # [5, 2, 1, 2]
#         weights = [max(n_visits) + 1 - v for v in n_visits]
#         probs = [w / sum(weights) for w in weights]
#         return np.random.choice(list(archive.values()), 1, p=probs)[0]

class Selector:
    def __init__(self):
        self.cells = DynamicArray(SIZE, dtype=object)
        self.weights = DynamicArray(SIZE, dtype=np.float32)
        # self.cells = []
        # self.weights = []

    def update_weight(self, index, n_visits):
        # self.weights[index] = 1 / np.sqrt((1 / self.weights[index]) ** 2 + 1)
        self.weights[index] = to_weight(n_visits)

    def add(self, cell):
        self.cells.append(cell)
        self.weights.append(to_weight(cell.visits))
        # self.weights.append(1 / np.sqrt(2))


class RouletteWheel(Selector):

    def __init__(self):
        super().__init__()

    def sample(self):
        probs = [w / sum(self.weights) for w in self.weights]
        return np.random.choice(self.cells.to_numpy(), 1, p=probs)[0]


class StochasticAcceptance(Selector):

    def __init__(self):
        super().__init__()
        # Weight is highest if cell visited only once.
        self.w_max = to_weight(1)

    def sample(self):
        i = np.random.randint(0, len(self.weights))
        p = self.weights[i] / self.w_max
        threshold = 1 - np.random.rand()
        if p > threshold:
            return self.cells[i]
        else:
            return self.sample()


class Uniform(Selector):

    def __init__(self):
        super().__init__()

    def sample(self):
        i = np.random.randint(0, len(self.weights))
        return self.cells[i]
