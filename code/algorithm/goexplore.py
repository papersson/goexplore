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
                 env, max_frames=4000000, seed=3533, verbose=True, logger=Logger(3533)):
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
        scores, n_cells, updates, discoveries = [], [], [], []
        while (self.n_frames < self.max_frames):
            # Sample cell from archive
            # print('START WHILE')
            cell = self.cell_selector.sample()
            # print('WHILE:', cell)
            self._explore_from(cell)
            # print()
            # self.highscore, self.n_frames = self._explore(
            #     cell, archive, self.highscore, self.n_frames, updates, discoveries)

            self.n_episodes += 1

            # Track for logging
            scores.append(self.highscore)
            n_cells.append(len(self.archive))

        # Extract cell that reached terminal state with highest score and smallest trajectory
        # cells = list(archive.values())
        cells = self.cell_selector.cells
        solved_cells = [cell for cell in cells if cell.done is True]
        best_cell = sorted(solved_cells)[
            0] if solved_cells else sorted(cells)[0]

        # Save logs
        duration = (time.time() - start)
        names = ['env', 'highscore', 'duration', 'episodes', 'n_frames',
                 'trajectory', 'scores', 'n_cells', 'discoveries', 'updates']
        values = [self.env.unwrapped.spec.id, self.highscore, str(timedelta(seconds=duration)), self.n_episodes,
                  self.n_frames, best_cell.trajectory, scores, n_cells, discoveries, updates]
        self.logger.add(names, values)
        self.logger.save()

    # def _explore_from(self, cell, archive, self.highscore, self.n_frames, updates, discoveries):
    def _explore_from(self, cell):
        # print('EXPLORE:', cell)
        # Restore to cell's simulator state and retrieve its score and trajectory
        # self.env.reset()
        cell.restore_state(self.env)
        trajectory, score = cell.history()

        # Termination criteria
        is_terminal = False
        n_steps = 0
        MAX_STEPS = 100

        # Only increment visits at most once per exploration iteration
        cells_seen_during_iteration = set()

        # n_updates, n_discoveries = 0, 0
        while (n_steps < MAX_STEPS and not is_terminal):
            # Interact
            action = self.agent.act()
            state, reward, is_terminal, _ = self.env.step(action)
            cell_representation = self.downsampler.process(state)

            # Track trajectory and score in case cells during current iteration need to be
            # updated or created
            trajectory.append(action)
            score += reward

            # Create cell if it is not in the archive. Update the cell if it has improved
            # performance, i.e. if it has a higher score or if it has the same score but
            # a shorter trajectory. A third alternative is that the current cell is not
            # a new cell nor is it a better cell, in which case nothing is done.
            cell = self._update_or_create_cell(
                self.archive, cell_representation, score, deepcopy(trajectory))
            # self.archive, cell_representation, score, deepcopy(trajectory), n_updates, n_discoveries)

            # Increment visit count/update weights if cell not seen during the episode
            if cell_representation not in cells_seen_during_iteration:
                cells_seen_during_iteration.add(cell_representation)
                cell_index = self.archive[cell_representation]
                cell.increment_visits()
                self.cell_selector.update_weight(cell_index, cell.visits)

            if score > self.highscore:
                self.highscore = score
                print(f'New highscore: {self.highscore}')

            n_steps += 1
            self.n_frames += 1
            if is_terminal:
                cell.set_done()
                break
            if self.verbose and self.n_frames % 50000 == 0:
                print(
                    f'Frames: {self.n_frames}\tScore: {self.highscore}\t Cells: {len(self.archive)}')
        # updates.append(n_updates)
        # discoveries.append(n_discoveries)

        # TODO: self.frame_update(), self.iteration_update()

        # return highscore, self.n_frames

    def _update_or_create_cell(self, archive, cell_representation, score, trajectory):
        # def _update_or_create_cell(self, archive, cell_representation, score, trajectory, n_updates, n_discoveries):
        if cell_representation in archive:
            # cell = archive[cell_representation]
            cell_index = archive[cell_representation]
            cell = self.cell_selector.cells[cell_index]
            if cell.is_worse(score, len(trajectory)):
                simulator_state = self.env.unwrapped.clone_state(
                    include_rng=True)
                cell.update(simulator_state, trajectory, score)
                self.cell_selector.update_weight(cell_index, cell.visits)
                # n_updates += 1
        else:
            simulator_state = self.env.unwrapped.clone_state(include_rng=True)
            cell = Cell(simulator_state, trajectory, score)
            self.cell_selector.add(cell)
            archive[cell_representation] = self.cell_index
            self.cell_index += 1
            # n_discoveries += 1
        return cell


class Cell:
    def __init__(self, simulator_state, trajectory=[], score=0.0):
        self.visits = 1
        self.done = False
        self.update(simulator_state, trajectory, score)

    def update(self, simulator_state, trajectory, score):
        self.simulator_state = simulator_state
        self.trajectory = trajectory
        self.score = score

    def increment_visits(self):
        self.visits += 1

    def get_weight(self):
        return 1 / np.log(self.visits + 1)

    def restore_state(self, env):
        env.unwrapped.restore_state(self.simulator_state)

    def history(self):
        return deepcopy(self.trajectory), deepcopy(self.score)

    def is_worse(self, score, actions_taken_length):
        return ((score > self.score)
                or (score == self.score and actions_taken_length < len(self.trajectory)))

    def set_done(self):
        self.done = True

    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={len(self.trajectory)}, visits={self.visits}, done={self.done})'

    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, len(self.trajectory)) < (-other.score, len(other.trajectory))
