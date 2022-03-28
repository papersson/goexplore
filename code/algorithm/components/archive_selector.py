import numpy as np
from utils.dynamicarray import DynamicArray


SIZE = 50000
def to_weight(n_visits): return 1 / np.sqrt(1 / n_visits + 1)


class Archive:
    def __init__(self, max_size):
        self.archive = {}
        self.insertion_index = 0
        self.cells = DynamicArray(SIZE, dtype=object)
        self.weights = DynamicArray(SIZE, dtype=np.float32)
        self.max_size = max_size

    def initialize(self, cell_repr, simulator_state):
        self.archive[cell_repr] = self.insertion_index
        self.add(cell_repr, (simulator_state,))

    def update_weight(self, cell_repr):
        if cell_repr not in self.archive:
            # Key error if archive size exceeded; ignore
            return
        cell_index = self.archive[cell_repr]
        cell = self.cells[cell_index]
        cell.increment_visits()
        self.weights[cell_index] = to_weight(cell.visits)

    def add(self, cell_repr, cell_state):
        if len(self.archive) >= self.max_size:
            # Don't add more cells if archive size exceeded
            return
        cell = Cell(*cell_state)
        self.archive[cell_repr] = self.insertion_index
        self.cells.append(cell)
        self.weights.append(to_weight(cell.visits))
        self.insertion_index += 1

    def get_best_cell(self):
        solved_cells = [cell for cell in self.cells if cell.done is True]
        best_cell = sorted(solved_cells)[
            0] if solved_cells else sorted(self.cells)[0]
        return best_cell

    def __getitem__(self, cell_repr):
        insertion_index = self.archive[cell_repr]
        return self.cells[insertion_index]

    def __contains__(self, key):
        return key in self.archive

    def __len__(self):
        return len(self.archive)


class RouletteWheel(Archive):

    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self):
        probs = [w / sum(self.weights) for w in self.weights]
        return np.random.choice(self.cells, 1, p=probs)[0]


class StochasticAcceptance(Archive):

    def __init__(self, max_size):
        super().__init__(max_size)
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


class Uniform(Archive):

    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self):
        i = np.random.randint(0, len(self.weights))
        return self.cells[i]


class Cell:
    def __init__(self, simulator_state, latest_action=None, traj_len=0, score=0.0):
        self.visits = 1
        self.done = False
        self.update(simulator_state, latest_action, traj_len, score)

    def update(self, simulator_state, latest_action, traj_len, score):
        self.simulator_state = simulator_state
        self.latest_action = latest_action
        self.traj_len = traj_len
        self.score = score

    def increment_visits(self):
        self.visits += 1

    def get_weight(self):
        return 1 / np.log(self.visits + 1)

    def load(self, env):
        env.unwrapped.restore_state(self.simulator_state)
        return self.latest_action, self.traj_len, self.score

    def should_update(self, score, traj_len):
        return ((score > self.score)
                or (score == self.score and traj_len < self.traj_len))

    def set_done(self):
        self.done = True

    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={self.traj_len}, visits={self.visits}, done={self.done})'

    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, self.traj_len) < (-other.score, self.traj_len)
