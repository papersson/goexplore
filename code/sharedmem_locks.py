# from __future__ import annotations
import gym
from dataclasses import dataclass, field
# from ale_py import ALEState
from typing import Any, List, Dict, Tuple
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
        # self.prev = prev
        self.action = action

    # def __repr__(self):
    #     return str(self.action)
        # return self.action_names[self.action]


@dataclass
class Cell:
    """ Class for tracking cell data. """
    score: float
    traj_len: int
    simulator_state: Any = field(repr=False)
    latest_action: ActionNode = None

    def update_if_better(self, score: float, traj_len: int) -> None:
        """ Cell should update if the current score is worse or if the current score is the same
        but the trajectory is shorter. """
        if (score > self.score) or (score == self.score and traj_len < self.traj_len):
            self.score = score
            self.traj_len = traj_len

    def load(self, env: Any) -> Tuple[ActionNode, float, int]:
        """ Restore to simulator state and return history. """
        env.unwrapped.restore_state(self.simulator_state)
        return self.latest_action, self.score, self.traj_len


def explore(archive, cells, end_index, lock, env, seed):
    print(seed)
    env.seed(np.random.randint(0, 2**32))
    env.action_space.seed(seed)
    np.random.seed(seed)
    n_iterations = 10

    for _ in range(n_iterations):
        cell = cells[np.random.randint(0, len(cells))]
        print(cell)
        latest_action, score, traj_len = cell.load(env)
        for _ in range(100):
            action = env.action_space.sample()
            state, reward, _, _ = env.step(action)

            cell_repr = downsample(state)
            latest_action = ActionNode(action, prev=latest_action)
            traj_len += 1
            score += reward

            lock.acquire()
            if cell_repr not in archive:
                archive[cell_repr] = end_index.value

                simulator_state = env.unwrapped.clone_state(include_rng=True)
                cell = Cell(score, traj_len, simulator_state, latest_action)
                end_index.value += 1
                cells.append(cell)
            else:
                cell_index = archive[cell_repr]
                cell = cells[cell_index]
                cell.update_if_better(score, traj_len)
            lock.release()
    print(n_iterations * 100)


if __name__ == "__main__":
    manager = mp.Manager()
    archive: Dict[Cell, int] = manager.dict()
    cells: List[Cell] = manager.list()
    end_index: int = mp.Value('i', 0)
    lock = mp.Lock()

    env = gym.make('PongDeterministic-v4')
    # ActionNode.action_names = env.unwrapped.get_action_meanings()

    state: np.ndarray = env.reset()
    cell_repr: Tuple = downsample(state)
    archive[cell_repr] = end_index.value
    end_index.value += 1

    simulator_state: Any = env.unwrapped.clone_state(include_rng=True)
    cell: Cell = Cell(0.0, 0, simulator_state)
    cells.append(cell)

    n_processes = mp.cpu_count()
    seeds = [np.random.randint(0, 2**32) for _ in range(n_processes)]
    processes = [mp.Process(target=explore, args=(
        archive, cells, end_index, lock, env, seeds[i]), daemon=True) for i in range(n_processes)]

    import time
    import sys
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(10000000)
    print(sys.getrecursionlimit())
    start = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    end = time.time()

    print(end-start)
    print(len(archive))
    print(len(cells))
    print(end_index)
    assert len(archive) == len(cells)
    assert end_index.value == len(cells)
    # Needs shared memory to modify archive, cells, and end_index. Needs locks to
    # synchronize properly (otherwise archive ends up smaller than cells). Seems to
    # work, but is slower than without multiprocessing.
