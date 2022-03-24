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


def downsample(state: np.ndarray) -> Tuple:
    width, height, num_colors = 12, 12, 8

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


@dataclass
class Cell:
    """ Class for tracking cell data. """
    insertion_index: int
    score: float
    traj_len: int
    simulator_state: Any = field(repr=False)
    latest_action: ActionNode = None

    def should_update(self, score: float, traj_len: int) -> bool:
        """ Cell should update if the current score is worse or if the current score is the same
        but the trajectory is shorter. """
        return (score > self.score) or (score == self.score and traj_len < self.traj_len)

    def update(self, score: float, traj_len: int, simulator_state: Any, latest_action: ActionNode) -> None:
        self.score = score
        self.traj_len = traj_len
        self.simulator_state = simulator_state
        self.latest_action = latest_action

    def load(self, env: Any) -> Tuple[ActionNode, float, int]:
        """ Restore to simulator state and return history. """
        env.unwrapped.restore_state(self.simulator_state)
        return self.latest_action, self.score, self.traj_len

    def get_trajectory(self):
        actions = []
        a = self.latest_action
        while a:
            actions = [a.action] + actions  # Prepend previous actions
            a = a.prev
        return actions


def explore(env, MAX_FRAMES, archive, cells, insertion_index):
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    t = 0
    for current_iteration in tqdm(range(MAX_FRAMES // 100)):
        cell = cells[np.random.randint(0, len(cells))]
        # print(t)
        # print(cell)
        latest_action, score, traj_len = cell.load(env)
        for _ in range(100):
            t += 1
            action = env.action_space.sample()
            state, reward, _, _ = env.step(action)

            cell_repr = downsample(state)
            latest_action = ActionNode(action, prev=latest_action)
            traj_len += 1
            score += reward

            if cell_repr not in archive:
                archive[cell_repr] = insertion_index
                cell = Cell(insertion_index, score, traj_len,
                            simulator_state, latest_action)
                insertion_index += 1
                # print(cell)
                cells.append(cell)
            else:
                cell_index = archive[cell_repr]
                cell = cells[cell_index]
                if cell.should_update(score, traj_len):
                    cell.update(score, traj_len,
                                simulator_state, latest_action)
    return insertion_index


if __name__ == "__main__":
    archive: Dict[Cell, int] = dict()
    cells: List[Cell] = list()
    insertion_index: int = 0

    env = gym.make('PongDeterministic-v4')

    # ActionNode.action_names = env.unwrapped.get_action_meanings()

    # Initialize archive
    state: np.ndarray = env.reset()
    cell_repr: Tuple = downsample(state)
    archive[cell_repr] = insertion_index

    simulator_state: Any = env.unwrapped.clone_state(include_rng=True)
    cell: Cell = Cell(insertion_index, 0.0, 0, simulator_state)
    print(cell)
    cells.append(cell)
    insertion_index += 1

    # MAX_FRAMES = 300
    MAX_FRAMES = 100000
    insertion_index = explore(env, MAX_FRAMES, archive, cells, insertion_index)

    print('Archive size:', len(archive))
    print('Cells array size:', len(cells))
    max_score = 0
    for cell in cells:
        # print(cell)
        if cell.score > max_score:
            max_score = cell.score
            best_cell = cell
    print(max_score)
    print(best_cell)
    traj = best_cell.get_trajectory()
    with open('mp_traj.npy', 'wb') as f:
        np.save(f, np.array(traj))

    assert len(archive) == len(cells)
    assert len(archive) == insertion_index

    # Fast, but doesn't perform very well. Reaches scores of 2-3 on Pong after 10M iterations,
    # without multiprocessing Pong is solved (score of 21) after 3-4M iterations.
