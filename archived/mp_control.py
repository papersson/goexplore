from tqdm import trange
# from .algorithm.goexplore import MAX_FRAMES_PER_ITERATION
from utils.dynamicarray import DynamicArray
from tqdm import tqdm
from copy import deepcopy
from enum import Enum
import time
import gym
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional
import cv2
import numpy as np
import multiprocessing as mp
from algorithm.components.archive_selector import Uniform, StochasticAcceptance


def downsample(state: np.ndarray) -> Tuple:
    width, height, num_colors = 12, 8, 16

    # Convert to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Shrink
    state = cv2.resize(state, (width, height), interpolation=cv2.INTER_AREA)

    # Reduce pixel depth
    bits = np.log2(num_colors)
    diff = 8 - bits  # Assumes 2^8 = 256 original depth
    k = 2**diff
    state = k * (state // k)

    return tuple(state.flatten())


class ActionNode:
    """ Single action that references the previous action. Used to extract the
    full cell trajectory when the algorithm has terminated. """

    def __init__(self, action, prev=None):
        self.prev = prev
        self.action = action

    def __repr__(self):
        return str(self.action)
        # return self.action_names[self.action]


# @dataclass
# class Cell:
#     """ Class for tracking cell data. """
#     insertion_index: int
#     score: float
#     traj_len: int
#     simulator_state: Any = field(repr=False)
#     # latest_action: ActionNode = None
#     latest_action: List[int]

#     def should_update(self, score: float, traj_len: int) -> bool:
#         """ Cell should update if the current score is worse or if the current score is the same
#         but the trajectory is shorter. """
#         return (score > self.score) or (score == self.score and traj_len < self.traj_len)

#     def update(self, score: float, traj_len: int, simulator_state: Any, latest_action: ActionNode) -> None:
#         self.score = score
#         self.traj_len = traj_len
#         self.simulator_state = simulator_state
#         self.latest_action = latest_action

#     def load(self, env: Any) -> Tuple[ActionNode, float, int]:
#         """ Restore to simulator state and return history. """
#         env.unwrapped.restore_state(self.simulator_state)
#         return self.latest_action, self.score, self.traj_len

#     def get_trajectory(self):
#         actions = []
#         a = self.latest_action
#         while a:
#             actions = [a.action] + actions  # Prepend previous actions
#             a = a.prev
#         return actions
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

    def get_trajectory(self):
        actions = []
        a = self.latest_action
        while a:
            actions = [a.action] + actions  # Prepend previous actions
            a = a.prev
        return actions

    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={self.traj_len}, visits={self.visits}, done={self.done})'

    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, self.traj_len) < (-other.score, self.traj_len)


def explore_from(cell, env):

    latest_action, traj_len, score = cell.load(env)

    # Maintain seen set to only increment visits at most once per exploration run
    n_steps = 0
    # while (n_steps < MAX_FRAMES_PER_ITERATION):
    for t in range(MAX_FRAMES_PER_ITERATION):
        # Interact
        action = env.action_space.sample()
        state, reward, is_terminal, _ = env.step(action)

        # Track cell object state in case cell needs to be updated or added
        simulator_state = env.unwrapped.clone_state(
            include_rng=True)
        latest_action = ActionNode(action, prev=latest_action)
        traj_len += 1
        score += reward
        cell_state = (simulator_state, latest_action, traj_len, score)

        # Handle cell event. Cases:
        # Cell discovered: add to archive
        # Cell is better than archived cell: update cell in archive
        # Cell is not discovered or better: do nothing
        cell_representation = downsample(state)
        # print(cell_representation)
        # print('TIMESTEP', t)
        # print('Archive size:', len(archive))
        # print('Cell_repr in archive:', cell_representation in archive)
        # print('======')
        if cell_representation not in archive:
            archive.add(cell_representation, cell_state)
        else:
            cell = archive[cell_representation]
            if cell.should_update(score, traj_len):
                cell.update(*cell_state)

        n_steps += 1

        if is_terminal:
            cell.set_done()


if __name__ == "__main__":

    start = time.time()
    env = gym.make('PongDeterministic-v4')
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    # archive = StochasticAcceptance()
    archive = Uniform()
    MAX_FRAMES_PER_ITERATION = 100
    MAX_FRAMES = 100000

    # Add first cell to archive
    starting_state = env.reset()
    simulator_state = env.unwrapped.clone_state(include_rng=True)
    cell_representation = downsample(starting_state)
    archive.initialize(cell_representation, simulator_state)

    scores, n_cells, iter_durations = [], [], []
    with trange(int(MAX_FRAMES / MAX_FRAMES_PER_ITERATION)) as t:
        for i in t:
            iter_start = time.time()
            # Progress bar
            t.set_description(f'Iteration {i}')
            t.set_postfix(num_cells=len(archive),
                          frames=(i+1) * MAX_FRAMES_PER_ITERATION)

            # Sample cell from archive
            cell = archive.sample()
            # print(cell)
            cell.increment_visits()
            explore_from(cell, env)

    # Extract cell that reached terminal state with highest score and smallest trajectory
    best_cell = archive.get_best_cell()

    print('Archive size:', len(archive))
    # print('Cells array size:', len(cells))
    # # max_score = -100000
    # for cell in sorted(cells)[:10]:
    #     print(cell)
    # # best_cell = cells[-1]
    # # print(max_score)
    print(best_cell)
    # # traj = best_cell.latest_action
    # traj = best_cell.get_trajectory()
    # print(len(traj))
    # with open('mp_traj.npy', 'wb') as f:
    #     np.save(f, np.array(traj))

    # assert len(archive) == len(cells)
    # assert len(archive) == insertion_index

    # Fast, but doesn't perform very well. Reaches scores of 2-3 on Pong after 10M iterations,
    # without multiprocessing Pong is solved (score of 21) after 3-4M iterations.


SIZE = 50000
# MAX_SIZE = 200000
def to_weight(n_visits): return 1 / np.sqrt(1 / n_visits + 1)


class Archive:
    def __init__(self):
        self.archive = {}
        self.insertion_index = 0
        self.cells = DynamicArray(SIZE, dtype=object)
        self.weights = DynamicArray(SIZE, dtype=np.float32)
        # self.cells = []
        # self.weights = []

    def initialize(self, cell_repr, simulator_state):
        self.archive[cell_repr] = self.insertion_index
        self.add(cell_repr, (simulator_state,))

    def update_weight(self, cell_repr):
        cell_index = self.archive[cell_repr]
        cell = self.cells[cell_index]
        cell.increment_visits()
        self.weights[cell_index] = to_weight(cell.visits)

    def add(self, cell_repr, cell_state):
        cell = Cell(*cell_state)
        self.archive[cell_repr] = self.insertion_index
        self.cells.append(cell)
        self.weights.append(to_weight(cell.visits))
        self.insertion_index += 1

    def get_best_cell(self):
        # solved_cells = [cell for cell in self.cells if cell.done is True]
        best_cell = sorted(self.cells)[0]
        # [
        # 0] if solved_cells else sorted(self.cells)[0]
        return best_cell

    def __getitem__(self, cell_repr):
        insertion_index = self.archive[cell_repr]
        return self.cells[insertion_index]

    def __contains__(self, key):
        return key in self.archive

    def __len__(self):
        return len(self.archive)
        # if key in self.archive:
        #     return self.archive[key]
        # else:
        #     raise KeyError(f"Cell not in archive")


class RouletteWheel(Archive):

    def __init__(self):
        super().__init__()

    def sample(self):
        probs = [w / sum(self.weights) for w in self.weights]
        return np.random.choice(self.cells, 1, p=probs)[0]


np.random.seed(0)


class StochasticAcceptance(Archive):

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


class Uniform(Archive):

    def __init__(self):
        super().__init__()

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

    def get_trajectory(self):
        actions = []
        a = self.latest_action
        while a:
            actions = [a.action] + actions  # Prepend previous actions
            a = a.prev
        return actions

    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={self.traj_len}, visits={self.visits}, done={self.done})'

    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, self.traj_len) < (-other.score, self.traj_len)
