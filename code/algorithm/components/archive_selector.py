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


class RouletteWheel:
    def initialize(self, cell):
        self.cells = [cell]
        self.weights = [1 / np.sqrt(2)]

    def update_weight(self, index):
        self.weights[index] = 1 / np.sqrt((1 / self.weights[index]) ** 2 + 1)

    def add_cell(self, cell):
        self.cells.append(cell)
        self.weights.append(1 / np.sqrt(2))

    def sample(self):
        probs = [w / sum(self.weights) for w in self.weights]
        return np.random.choice(self.cells, 1, p=probs)[0]


class StochasticAcceptance:
    # def sample(self, archive):
    #     visits = [cell.visits for cell in archive.values()]
    #     weights = [1 / np.log(c.visits + 1) for c in archive.values()]
    #     probs = [w / sum(weights) for w in weights]
    #     return np.random.choice(list(archive.values()), 1, p=probs)[0]
    # def __init__(self):
    # self.cells = []

    def initialize(self, cell):
        self.cells = [cell]
        self.weights = [1 / np.sqrt(2)]
        self.w_max = 1 / np.sqrt(2)

    # def update(self, cells):
    #     self.cells = cells
    #     weights = np.array([c.get_weight() for c in cells])
    #     self.probs = weights / sum(weights)

    def update_weight(self, index):
        # Increases visits by 1 and updates weight accordingly
        # Maybe inefficient? Alternative: separate visits and weights lists
        self.weights[index] = 1 / np.sqrt((1 / self.weights[index]) ** 2 + 1)

    def add_cell(self, cell):
        # Append cell to the end of cells, and append init weight to end of weights
        self.cells.append(cell)
        self.weights.append(1 / np.sqrt(2))

    def sample(self):
        i = np.random.randint(0, len(self.weights))
        p = self.weights[i] / self.w_max
        threshold = 1 - np.random.rand()
        if p > threshold:
            return self.cells[i]
        else:
            return self.sample()
        # return self.cells[i]
        # visits = [cell.visits for cell in archive.values()]
        # weights = [1 / np.log(c.visits + 1) for c in archive.values()]
        # probs = [w / sum(weights) for w in weights]

        # return np.random.choice(self.cells, 1, p=self.probs)[0]
