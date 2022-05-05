import itertools
import numpy as np


# Novelty: parameters and function that converts counts to a weight.
w_selects = 0.1
w_visits = 0.3

eps1 = 0.001
eps2 = 0.00001


def subscore(w, count):
    return w * np.sqrt(1 / (count + eps1)) + eps2


def to_weight(n_selects, n_visits):
    return subscore(w_selects, n_selects) + subscore(w_visits, n_visits)


class Cell:
    """ Cell object.

    A cell stores its score, novelty weight, trajectory, and simulator state.

    Attributes
    ----------
    simulator_state : ALEState
        Simulator state.

    prev : Node
        Reference to predecessor node.

    actions : array
        Array of actions connecting predecessor to current node.

    traj_len : int
        Length of trajectory.

    score : float
        Game score.

    insertion_index : int
        The cell's insertion index.

    n_selected : int
        The number of times the cell has been selected.

    n_visits : int
        The number of times the cell has been visited.
    """

    id_iter = itertools.count()

    def __init__(self, simulator_state, prev=None, actions=None, traj_len=0, score=0.0):
        self.insertion_index = next(self.id_iter)
        self.n_selected = 0
        self.n_visits = 1
        self.update(simulator_state, prev, actions, traj_len, score)

    def update(self, simulator_state, prev, actions, traj_len, score):
        """ Update cell state. """
        self.simulator_state = simulator_state
        self.prev = prev
        self.actions = actions
        self.traj_len = traj_len
        self.score = score
        self.n_selected = 1  # Reset when cell is updated

    def load(self, env):
        """ Restore environment to cell's state. """
        env.unwrapped.restore_state(self.simulator_state)
        return self.traj_len, self.score

    def should_update(self, score, traj_len):
        """ Return true if current cell scores higher, or if it has
        the same score but a shorter trajectory.
        """
        return ((score > self.score)
                or (score == self.score and traj_len < self.traj_len))

    def __repr__(self):
        return f'Cell(id={self.insertion_index}, score={self.score}, traj_len={self.traj_len}, n_selected={self.n_selected}, n_visits={self.n_visits})'

    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, self.traj_len) < (-other.score, self.traj_len)


class Archive:
    """ Cell archive.

    The archive adds and updates encountered cells, and samples cells
    proportianlly to their novelty.

    Parameters
    ----------
    max_size : int
        The maximum number of cells in the archive.

    Attributes
    ----------
    table : dict
        Dictionary that maps cell tuple to insertion index.

    cells : list
        List of Cell objects.

    insertion_index : int
        Incrementally increasing cell insertion index.

    weights : list
        List of cell weights.

    """

    def __init__(self, max_size):
        self.table = {}
        self.insertion_index = 0
        self.cells = []
        self.weights = []
        self.max_size = max_size
        Cell.id_iter = itertools.count()

    def initialize(self, cell_repr, simulator_state):
        """ Initialize archive with starting cell.

        Parameters
        ----------
        cell_repr : tuple
            Tuple of cell pixels.

        simulator_state : ALEState
            Simulator state.

        """
        self.add(cell_repr, (simulator_state,))

    def add(self, cell_repr, cell_state):
        """ Add cell to the archive.

        Add the cell representation to the lookup table and increment the
        insertion index. Also create a cell object, and append it and its
        weight to the cell and weight lists.

        Parameters
        ----------
        cell_repr : tuple
            Tuple of cell pixels.

        cell_state : tuple
            Tuple containing the cell state, which can contain simulator state,
            a reference to the previous node, an array of actions, the trajectory
            length, and the game score.

        """
        if len(self.table) >= self.max_size:
            # Don't add more cells if archive size exceeded
            return
        self.table[cell_repr] = self.insertion_index
        self.insertion_index += 1

        cell = Cell(*cell_state)
        self.cells.append(cell)
        self.weights.append(to_weight(cell.n_selected, cell.n_visits))
        return cell

    def update(self, cell, cell_state):
        """ Update a cell in the archive.

        Parameters
        ----------
        cell : Cell
            Cell object.

        cell_state : tuple
            Tuple containing the cell state, which can contain simulator state,
            a reference to the previous node, an array of actions, the trajectory
            length, and the game score.

        """
        cell.update(*cell_state)
        return cell

    def increment_visits(self, cell_repr):
        """ Increment visit count for cell object.

        Parameters
        ----------
        cell_repr : tuple
            Tuple representing the cell pixels.
        """

        cell = self.__getitem__(cell_repr)
        cell.n_visits += 1
        self.weights[cell.insertion_index] = to_weight(
            cell.n_selected, cell.n_visits)

    def get_best_cell(self):
        best_cell = sorted(self.cells)[0]
        return best_cell

    def get_best_trajectory(self):
        """ Get higher scoring cell and recover its trajectory. """
        current_cell = self.get_best_cell()
        actions = []
        node = current_cell.prev
        while node:
            actions = node.actions + actions
            node = node.prev
        return actions

    def __getitem__(self, cell_repr):
        insertion_index = self.table[cell_repr]
        return self.cells[insertion_index]

    def __contains__(self, key):
        return key in self.table

    def __len__(self):
        return len(self.table)


class RouletteWheel(Archive):

    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self):
        """ Sample cell using roulette wheel selection. """
        probs = self.weights / np.sum(self.weights)
        cell = np.random.choice(self.cells, 1, p=probs)[0]
        cell.n_selected += 1
        return cell


class StochasticAcceptance(Archive):

    def __init__(self, max_size):
        super().__init__(max_size)
        # Weight is highest if cell visited only once and expanded zero times.
        self.w_max = to_weight(0, 1)

    def sample(self):
        """ Sample cell using stochastic acceptance selection. """
        i = np.random.randint(0, len(self.weights))
        p = self.weights[i] / self.w_max
        threshold = 1 - np.random.rand()
        if p > threshold:
            cell = self.cells[i]
            cell.n_selected += 1
            return cell
        else:
            return self.sample()


class Uniform(Archive):

    def __init__(self, max_size):
        super().__init__(max_size)

    def sample(self):
        """ Randomly sample a cell. """
        i = np.random.randint(0, len(self.weights))
        cell = self.cells[i]
        cell.n_selected += 1
        return cell
