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
    def __init__(self, agent, downsampler, archive_selector,
                 max_frames=4000000, game='Pong', seed=3533, verbose=True, logger=Logger(3533)):
        # Set game and seed
        self.agent = agent
        self.downsampler = downsampler
        self.archive_selector = archive_selector
        self.max_frames = max_frames
        self.game = game
        self.seed = seed
        self.verbose = verbose
        self.logger = logger
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env = gym.make(f'{game}Deterministic-v4')
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        # Initialize algorithm
    def run(self):
        start = time.time()
        state = self.env.reset()

        simulator_state = self.env.unwrapped.clone_state(include_rng=True)
        actions_taken = []
        score = 0.0
        cell = Cell(simulator_state, actions_taken, score)
        cell.increment_visits()

        # Create archive and add initial cell
        archive = {}
        cell_representation = self.downsampler.process(state)
        archive[cell_representation] = cell

        highscore, n_frames = 0, 0
        scores, n_cells = [], []
        # while (n_frames < self.max_frames):
        for _ in tqdm(range(int(self.max_frames / 100))):
            # Sample cell from archive
            cell = self.archive_selector.sample(archive)
            highscore, n_frames = self._explore(
                cell, archive, highscore, n_frames)

            # Track for logging
            scores.append(highscore)
            n_cells.append(len(archive))
            if self.verbose and n_frames % 10000 == 0:
                print(
                    f'Frames: {n_frames}\tScore: {highscore}\t Cells: {len(archive)}')

        # Extract cell that reached terminal state with highest score and smallest trajectory
        cells = list(archive.values())
        solved_cells = [cell for cell in cells if cell.done is True]
        best_cell = sorted(solved_cells)[
            0] if solved_cells else sorted(cells)[0]

        # Save logs
        duration = (time.time() - start)
        names = ['highscore', 'duration', 'n_frames',
                 'action_history', 'scores', 'n_cells']
        values = [highscore, str(timedelta(seconds=duration)),
                  n_frames, best_cell.action_history, scores, n_cells]
        self.logger.add(names, values)
        self.logger.save()

    def _explore(self, cell, archive, highscore, n_frames):
        self.env.reset()
        cell.restore_state(self.env)
        action_history, score = cell.history()

        is_terminal = False
        n_steps = 0
        seen_cells_during_episode = set()

        MAX_STEPS = 100
        while (n_steps < MAX_STEPS and not is_terminal):
            # TODO: n_steps and sticky action configurable?
            # Interact
            action = self.agent.act()
            state, reward, is_terminal, _ = self.env.step(action)
            action_history.append(action)
            score += reward
            if score > highscore:
                highscore = score
                print(f'New highscore: {highscore}')

            # Update or add cell to archive
            cell_representation = self.downsampler.process(state)
            cell = self._update_or_create_cell(
                archive, cell_representation, score, deepcopy(action_history))

            # Increment visit count if cell not seen during the episode
            if cell_representation not in seen_cells_during_episode:
                seen_cells_during_episode.add(cell_representation)
                cell.increment_visits()

            n_steps += 1
            n_frames += 1
            if is_terminal:
                cell.set_done()
                break

        return highscore, n_frames

    def _update_or_create_cell(self, archive, cell_representation, score, action_history):
        if cell_representation in archive:
            cell = archive[cell_representation]
            if cell.is_worse(score, len(action_history)):
                simulator_state = self.env.unwrapped.clone_state(
                    include_rng=True)
                cell.update(simulator_state, action_history, score)
        else:
            simulator_state = self.env.unwrapped.clone_state(include_rng=True)
            cell = Cell(simulator_state, action_history, score)
            archive[cell_representation] = cell
        return cell


class Cell:
    def __init__(self, simulator_state, action_history, score):
        self.visits = 0
        self.done = False
        self.update(simulator_state, action_history, score)

    def update(self, simulator_state, action_history, score):
        self.simulator_state = simulator_state
        self.action_history = action_history
        self.score = score

    def increment_visits(self):
        self.visits += 1

    def restore_state(self, env):
        env.unwrapped.restore_state(self.simulator_state)

    def history(self):
        return deepcopy(self.action_history), deepcopy(self.score)

    def is_worse(self, score, actions_taken_length):
        return ((score > self.score)
                or (score == self.score and actions_taken_length < len(self.action_history)))

    def set_done(self):
        self.done = True

    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={len(self.action_history)}, visits={self.visits}, done={self.done})'

    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, len(self.action_history)) < (-other.score, len(other.action_history))
