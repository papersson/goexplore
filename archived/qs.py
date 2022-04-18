# from __future__ import annotations
from copy import deepcopy
from enum import Enum
import time
import gym
from dataclasses import dataclass, field
# from ale_py import ALEState
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


Command = Enum('Command', 'DISCOVERY UPDATE READY DONE')


@dataclass
class Message:
    command: Command
    data: Optional[Tuple]


def explore(in_q, out_q, env, seed, MAX_FRAMES):
    env.seed(np.random.randint(0, 2**32))
    env.action_space.seed(seed)
    np.random.seed(seed)

    # while current_frame < MAX_FRAMES:
    for current_iteration in range(MAX_FRAMES // 100):
        out_q.put(Message(Command.READY, None))
        archive, cells = in_q.get()

        cell = cells[np.random.randint(0, len(cells))]
        # print(cell)
        latest_action, score, traj_len = cell.load(env)
        for _ in range(100):
            action = env.action_space.sample()
            state, reward, _, _ = env.step(action)

            cell_repr = downsample(state)
            latest_action = ActionNode(action, prev=latest_action)
            traj_len += 1
            score += reward

            if cell_repr not in archive:
                out_q.put(Message(Command.DISCOVERY,
                                  (cell_repr, score, traj_len, simulator_state, latest_action)))
            else:
                cell_index = archive[cell_repr]
                cell = cells[cell_index]
                if cell.should_update(score, traj_len):
                    out_q.put(
                        Message(Command.UPDATE, (cell, score, traj_len, simulator_state, latest_action)))
    out_q.put(Message(Command.DONE, None))


if __name__ == "__main__":
    archive: Dict[Cell, int] = dict()
    cells: List[Cell] = list()
    end_index: int = 0

    env = gym.make('PongDeterministic-v4')
    # ActionNode.action_names = env.unwrapped.get_action_meanings()

    state: np.ndarray = env.reset()
    cell_repr: Tuple = downsample(state)
    archive[cell_repr] = end_index
    end_index += 1

    simulator_state: Any = env.unwrapped.clone_state(include_rng=True)
    cell: Cell = Cell(0.0, 0, simulator_state)
    cells.append(cell)

    in_q = mp.SimpleQueue()
    out_q = mp.SimpleQueue()
    n_processes = mp.cpu_count()
    seeds = [np.random.randint(0, 2**32) for _ in range(n_processes)]
    max_frames = 1000000 // n_processes
    processes = [mp.Process(target=explore, args=(in_q, out_q, deepcopy(
        env), seeds[i], max_frames), daemon=True) for i in range(n_processes)]
    start = time.time()
    for p in processes:
        # in_q.put((archive, cells))
        p.start()
    dones = []
    while len(dones) < 4:
        # while True:
        msg: Message = out_q.get()

        if msg.command == Command.READY:
            in_q.put((archive, cells))
            # print('READY')
        elif msg.command == Command.DISCOVERY:
            # print('DISCOVERY')
            cell_repr, score, traj_len, simulator_state, latest_action = msg.data
            if cell_repr not in archive:
                archive[cell_repr] = end_index
                end_index += 1
                cell = Cell(score, traj_len, simulator_state, latest_action)
                cells.append(cell)
        elif msg.command == Command.UPDATE:
            # print('UPDATE')
            cell, score, traj_len, simulator_state, latest_action = msg.data
            if cell.should_update(score, traj_len):
                cell.update(score, traj_len, simulator_state, latest_action)
        elif msg.command == Command.DONE:
            dones.append('done')
        else:
            raise NotImplementedError
    end = time.time()
    print(end-start)
    for p in processes:
        p.join()

    print(len(archive))
    print(len(cells))
    print(end_index)

    assert len(archive) == len(cells)
    assert len(archive) == end_index
