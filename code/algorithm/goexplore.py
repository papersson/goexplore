from tqdm import tqdm
from utils.logger import Logger
import numpy as np
import random
import gym
import time
from datetime import timedelta
from copy import deepcopy
import sys
sys.path.append('..')


class GoExplore:
    def __init__(self, agent, downsampler, cell_selector,
                 env, logger, max_frames=4000000, seed=3533, verbose=True):
        # Set seed, environment/game, agent, downsampler, and archive selector
        self.seed = seed
        self.env = env
        self.agent = agent
        self.downsampler = downsampler
        self.cell_selector = cell_selector
        self.max_frames = max_frames
        self.verbose = verbose
        self.logger = logger
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        # Track during run
        self.archive = {}
        self.cell_index = 0
        self.highscore, self.n_frames, self.n_episodes = 0, 0, 0

    def run(self):
        start = time.time()

        # Initialize archive
        starting_state = self.env.reset()
        simulator_state = self.env.unwrapped.clone_state(include_rng=True)
        cell_representation = self.downsampler.process(starting_state)
        # Maps from cell tuple representation to insertion index
        # archive = {cell_representation: self.cell_index}
        self.archive[cell_representation] = self.cell_index
        self.cell_index += 1

        # Add starting state cell to cell selector
        cell = Cell(simulator_state)
        self.cell_selector.add(cell)

        # self.highscore, self.n_frames, self.n_episodes = 0, 0, 0
        scores, n_cells, iter_durations, steps_in_iterations = [], [], [], []
        while (self.n_frames < self.max_frames):
            print('Processed frames', self.n_frames, end='\r')
            iter_start = time.time()
            # Sample cell from archive
            cell = self.cell_selector.sample()
            steps_in_iteration = self._explore_from(cell)

            self.n_episodes += 1

            # Track for logging
            iter_end = time.time()
            scores.append(self.highscore)
            n_cells.append(len(self.archive))
            iter_durations.append(round(iter_end - iter_start, 3))
            steps_in_iterations.append(steps_in_iteration)
            print(
                f'Processed frames: {self.n_frames}/{self.max_frames}\tIteration time: {round(iter_end - iter_start, 3)}s\tNum cells: {len(self.cell_selector.cells)}', end='\r')

        # Extract cell that reached terminal state with highest score and smallest trajectory
        # cells = list(archive.values())
        cells = self.cell_selector.cells
        solved_cells = [cell for cell in cells if cell.done is True]
        best_cell = sorted(solved_cells)[
            0] if solved_cells else sorted(cells)[0]

        # Save logs
        duration = (time.time() - start)
        if self.logger:
            names = ['highscore', 'duration', 'n_frames',
                     'trajectory', 'scores', 'n_cells', 'iter_durations']
            values = [self.highscore, str(timedelta(seconds=duration)),
                      self.n_frames, best_cell.get_trajectory(), scores, n_cells, iter_durations]
            self.logger.add(names, values)
            self.logger.save()

    # def _explore_from(self, cell, archive, self.highscore, self.n_frames, updates, discoveries):
    def _explore_from(self, cell):
        # Restore to cell's simulator state and retrieve its score and trajectory
        latest_action, traj_len, score = cell.load(self.env)

        # Termination criteria
        is_terminal = False
        n_steps = 0
        MAX_STEPS = 100

        # Maintain seen set to only increment visits at most once per exploration iteration
        cells_seen_during_iteration = set()

        while (n_steps < MAX_STEPS):
            # Interact
            action = self.agent.act()
            # self.env.render()
            state, reward, is_terminal, _ = self.env.step(action)
            cell_representation = self.downsampler.process(state)

            # Track trajectory and score in case cells during current iteration need to be
            # updated or created
            latest_action = Action(action, prev=latest_action)
            traj_len += 1
            score += reward

            # Create cell if it is not in the archive. Update the cell if it is in the archive,
            # but has improved performance, i.e. if it has a higher score or if it has the same
            # score but a shorter trajectory. A third alternative is that the current cell is
            # not a new cell nor is it a better cell, in which case nothing is done.
            if cell_representation not in self.archive:
                simulator_state = self.env.unwrapped.clone_state(
                    include_rng=True)
                cell = Cell(simulator_state, latest_action, traj_len, score)
                self.cell_selector.add(cell)
                self.archive[cell_representation] = self.cell_index
                self.cell_index += 1
            else:
                # Get cell from archive and compare scores/traj_lens with current cell
                cell_index = self.archive[cell_representation]
                cell = self.cell_selector.cells[cell_index]

                if cell.is_worse(score, traj_len):
                    simulator_state = self.env.unwrapped.clone_state(
                        include_rng=True)
                    cell.update(simulator_state,
                                latest_action, traj_len, score)
                    self.cell_selector.update_weight(cell_index, cell.visits)
                # TODO: Do I need to clear action links from memory if the cell is not updated?

            # Increment visit count/update weights if cell not seen during the episode
            if cell_representation not in cells_seen_during_iteration:
                cells_seen_during_iteration.add(cell_representation)
                cell_index = self.archive[cell_representation]
                cell.increment_visits()
                self.cell_selector.update_weight(cell_index, cell.visits)

            if score > self.highscore:
                self.highscore = score
                # print(f'New highscore: {self.highscore}')

            n_steps += 1
            self.n_frames += 1
            # print(
            #     f'Iterations: {self.n_episodes}\tSteps: {n_steps} \t n_steps < MAX_STEPS: {n_steps < MAX_STEPS} \t is_terminal: {is_terminal}', end='\r')
            # if self.n_frames > 100000 - 150:
            #     time.sleep(1)
            if is_terminal:
                cell.set_done()
                # break
            # if self.verbose and self.n_frames % 50000 == 0:
            #     print(
            #         f'Frames: {self.n_frames}\tScore: {self.highscore}\t Cells: {len(self.archive)}')
        return n_steps


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

    def is_worse(self, score, traj_len):
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


class Action:
    def __init__(self, action, prev=None):
        self.prev = prev
        self.action = action

    def __repr__(self):
        return str(self.action)
