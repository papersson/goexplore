from copy import deepcopy
import gym
import random
import numpy as np
from cell import Cell
from tqdm import tqdm


class GoExplore:
    def __init__(self, agent, downsampler, archive_selector,
                 max_frames=4000000, game='Pong', seed=3533, verbose=True):
        # Set game and seed
        self.agent = agent
        self.downsampler = downsampler
        self.archive_selector = archive_selector
        self.max_frames = max_frames
        self.game = game
        self.seed = seed
        self.verbose = verbose
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env = gym.make(f'{game}Deterministic-v4')
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        # Initialize algorithm
    def run(self):
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
        # while (n_frames < self.max_frames):
        for _ in tqdm(range(int(self.max_frames / 100))):
            # Sample cell from archive
            cell = self.archive_selector.sample(archive)
            highscore = self._explore(cell, archive, highscore)
            n_frames += 100

            if self.verbose and n_frames % 100000 == 0:
                print(
                    f'Frames: {n_frames}\tScore: {highscore}\t Cells: {len(archive)}')

    def _explore(self, cell, archive, highscore):
        self.env.reset()
        cell.restore_state(self.env)
        action_history, score = cell.history()

        is_terminal = False
        n_steps = 0
        seen_cells_during_episode = set()

        action = self.agent.act()
        MAX_STEPS, STICKY_PROB = 100, 0.25
        while (n_steps < MAX_STEPS and not is_terminal):
            # TODO: n_steps and sticky action configurable?
            # Interact
            if random.random() > STICKY_PROB:  # Sticky actions
                action = self.agent.act()
            state, reward, is_terminal, _ = self.env.step(action)
            action_history.append(action)
            score += reward
            if score > highscore:
                highscore = score

            # Update or add cell to archive
            cell_representation = self.downsampler.process(state)
            cell = self._update_or_create_cell(
                archive, cell_representation, score, deepcopy(action_history))

            # Increment visit count if cell not seen during the episode
            if cell_representation not in seen_cells_during_episode:
                seen_cells_during_episode.add(cell_representation)
                cell.increment_visits()

            n_steps += 1
            if is_terminal:
                cell.set_done()
                break

        return highscore

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
