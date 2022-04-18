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


Command = Enum('Command', 'DISCOVERY UPDATE READY DONE')


@dataclass
class Message:
    command: Command
    data: Optional[Tuple]
    timestep: int


def explore(parent, out_q, env, seed, MAX_FRAMES, archive, cells):
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    # seed = 0
    # env.seed(seed)
    # env.action_space.seed(seed)
    # np.random.seed(seed)
    t = 0

    # while current_frame < MAX_FRAMES:
    from tqdm import tqdm
    for current_iteration in tqdm(range(MAX_FRAMES // 100)):
        # for current_iteration in tqdm(range(MAX_FRAMES // 100)):
        # print(len(cells))

        # out_q.put(Message(Command.READY, None, t))
        # parent.get()
        cell = cells[np.random.randint(0, len(cells))]
        # print(cell)
        latest_action, score, traj_len = cell.load(env)
        for _ in range(100):
            t += 1
            action = env.action_space.sample()
            state, reward, _, _ = env.step(action)

            simulator_state = env.unwrapped.clone_state(include_rng=True)
            cell_repr = downsample(state)
            latest_action = ActionNode(action, prev=latest_action)
            traj_len += 1
            score += reward

            if cell_repr not in archive:
                out_q.put(Message(Command.DISCOVERY,
                                  (cell_repr, score, traj_len,
                                   simulator_state, latest_action),
                                  t
                                  ))
                # parent.get()
            else:
                cell_index = archive[cell_repr]
                cell = cells[cell_index]
                if cell.should_update(score, traj_len):
                    # if score > 0:
                    #     print(t)
                    #     print(score)
                    out_q.put(
                        Message(Command.UPDATE, (cell_index, score, traj_len, simulator_state, latest_action), t))
                    # parent.get()
    out_q.put(Message(Command.DONE, None, t))


archive = {}
cells = []
if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(1000000000)
    with mp.Manager() as manager:
        archive: Dict[Cell, int] = manager.dict()
        cells: List[Cell] = manager.list()
        insertion_index: int = 0

        env = gym.make('PongDeterministic-v4')
        # ActionNode.action_names = env.unwrapped.get_action_meanings()

        # Initialize archive
        state: np.ndarray = env.reset()
        cell_repr: Tuple = downsample(state)
        archive[cell_repr] = insertion_index

        simulator_state: Any = env.unwrapped.clone_state(include_rng=True)
        cell: Cell = Cell(insertion_index, 0.0, 0, simulator_state)
        cells.append(cell)
        insertion_index += 1

        # Initialize processes; multiple producers, single consumer
        out_q = mp.Queue()
        parent = mp.Queue()
        n_processes = 1
        # n_processes = mp.cpu_count() * 2
        seeds = [np.random.randint(0, 2**32) for _ in range(n_processes)]
        # max_frames = 300
        max_frames = int(100000 // n_processes)
        processes = [mp.Process(target=explore, args=(parent, out_q, deepcopy(
            env), seeds[i], max_frames, archive, cells), daemon=True) for i in range(n_processes)]
        start = time.time()
        for p in processes:
            p.start()
        dones = []
        while len(dones) < n_processes:
            # while True:
            msg: Message = out_q.get()
            # print(insertion_index)

            # if msg.command == Command.READY:
            # parent.put((archive, cells))
            if msg.command == Command.DISCOVERY:
                # print('DISCOVERY')
                cell_repr, score, traj_len, simulator_state, latest_action = msg.data
                if cell_repr not in archive:
                    archive[cell_repr] = insertion_index
                    cell = Cell(insertion_index, score, traj_len,
                                simulator_state, latest_action)
                    insertion_index += 1
                    # print(cell)
                    cells.append(cell)
            elif msg.command == Command.UPDATE:
                # print('UPDATE')
                cell_index, score, traj_len, simulator_state, latest_action = msg.data
                # print(score)
                cell = cells[cell_index]
                if score >= 21:
                    break
                # cell should be retrieved from local cells
                t = msg.timestep
                if cell.should_update(score, traj_len):
                    # if t in range(200, 300):
                    #     print(t)
                    #     print(f'Before update: {cell}')
                    #     cells[cell_index] = Cell(
                    #         cell_index, score, traj_len, simulator_state, latest_action)
                    #     # cell.update(score, traj_len,
                    #     #             simulator_state, latest_action)
                    #     print(f'After update: {cell}')
                    # else:
                    cells[cell_index] = Cell(
                        cell_index, score, traj_len, simulator_state, latest_action)
                # parent.put(None)
            elif msg.command == Command.DONE:
                dones.append('done')
            else:
                raise NotImplementedError
            # parent.put(None)
        end = time.time()
        print('Time:', end-start)
        for p in processes:
            p.join()

        print('Archive size:', len(archive))
        print('Cells array size:', len(cells))
        max_score = 0
        for cell in cells[:50]:
            print(cell)
            if cell.score > max_score:
                max_score = cell.score
                best_cell = cell
        print(max_score)
        traj = best_cell.get_trajectory()
        with open('mp_traj.npy', 'wb') as f:
            np.save(f, np.array(traj))

        assert len(archive) == len(cells)
        assert len(archive) == insertion_index

        # Fast, but doesn't perform very well. Reaches scores of 2-3 on Pong after 10M iterations,
        # without multiprocessing Pong is solved (score of 21) after 3-4M iterations.
