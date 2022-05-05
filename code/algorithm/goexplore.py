""" Example docstring.

Some text with some more bla bla bla. What is this now? Huh?!

Attributes
-----------
N_EXPLORATION_STEPS : int
    Number of steps per exploration run.

"""


from configparser import Interpolation
from tqdm import trange
import numpy as np
import time
from datetime import timedelta
import cv2


N_EXPLORATION_STEPS = 100


class GoExplore:
    """ Simulator-based GoExplore.

    GoExplore selects a cell and then explores some number of iterations. It (optionally)
    saves the highest-scoring trajectory to "experiments/<experiment>.trajectory".

    Parameters
    ----------
    agent : Agent
        This is some text.


    downsampler : Downsampler
        This is some text.

    archive : Archive
        This is another text

    env : ALE.kljaflkjf


    agent : Agent
        This is some text.

    Notes
    -----
    This is an implementation of the original version of GoExplore [1]_, modified
    to support stochastic acceptance selection [2]_.


    References
    ----------
    .. [1] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley,
        and Jeff Clune. Go-explore: a new approach for hard-exploration
        problems, 2019.

    .. [2] Adam Lipowski and Dorota Lipowska. Roulette-wheel se-
        lection via stochastic acceptance. Physica A: Statistical Me-
        chanics and its Applications, 391(6):2193–2196, 2012.

    """

    def __init__(self, agent, cell_params, archive,
                 env, seed, max_frames, logger):
        # Initialize hyperparameters.
        self.seed = seed
        self.env = env
        self.agent = agent
        # self.downsampler = self._downsampler(cell_params)
        self.cell_params = cell_params
        self.archive = archive
        self.max_frames = max_frames
        self.logger = logger
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        # Track during run for plotting.
        self.highscore, self.n_frames = 0, 0
        print(self.__doc__)

    def _downsample(self, img):
        width, height, depth = self.cell_params

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (width, height),
                         interpolation=cv2.INTER_AREA)

        bits = np.log2(depth)
        diff = 8 - bits
        k = 2 ** diff
        img = k * (img // k)
        return tuple(img.flatten())

    def run(self):
        start = time.time()

        # Initialize archive with starting cell.
        starting_state = self.env.reset()
        simulator_state = self.env.unwrapped.clone_state(include_rng=True)
        # cell_representation = self.downsampler.process(starting_state)
        cell_representation = self._downsample(starting_state)
        self.archive.initialize(cell_representation, simulator_state)

        # Track data
        scores_data = []
        n_cells_data = []
        n_updates_data = []
        n_discoveries_data = []
        iter_durations_data = []

        # Run GoExplore for self.max_frames frames
        n_iterations = int(self.max_frames / N_EXPLORATION_STEPS)
        with trange(n_iterations) as t:
            for i in t:
                iter_start = time.time()
                # Progress bar.
                t.set_description(f'Iteration {i}')
                t.set_postfix(highscore=self.highscore, num_cells=len(self.archive),
                              frames=(i+1) * N_EXPLORATION_STEPS)

                # Sample cell from archive and explore.
                cell = self.archive.sample()
                n_updates, n_discoveries = self._explore_from(cell)

                # Update data.
                iter_end = time.time()
                scores_data.append(self.highscore)
                n_cells_data.append(len(self.archive))
                n_updates_data.append(n_updates)
                n_discoveries_data.append(n_discoveries)
                iter_durations_data.append(round(iter_end - iter_start, 3))

        # Extract cell that reached terminal state with highest score and smallest trajectory
        best_cell = self.archive.get_best_cell()
        traj = best_cell.get_trajectory()
        swarm = [cell.traj_len for cell in self.archive.cells]

        # Save logging data.
        duration = (time.time() - start)
        if self.logger:
            names = ['highscore', 'traj_len', 'total_cells', 'duration', 'n_frames',
                     'trajectory', 'scores', 'n_cells', 'n_updates', 'n_discoveries', 'iter_durations', 'swarm']
            values = [self.highscore, len(traj), len(self.archive), str(timedelta(seconds=duration)),
                      self.n_frames, traj, scores_data, n_cells_data, n_updates_data, n_discoveries_data, iter_durations_data, swarm]
            self.logger.add(names, values)
            self.logger.save()

    def _explore_from(self, cell):
        # Track updates and discoveries for logging.
        n_updates, n_discoveries = 0, 0

        # Restore to cell's simulator state.
        traj_len, score = cell.load(self.env)
        actions = []
        prev = cell.prev

        # Explore for 100 steps.
        for _ in range(N_EXPLORATION_STEPS):
            # Interact.
            action = self.agent.act()
            state, reward, is_terminal, _ = self.env.step(action)

            # Track cell state in case cell needs to be updated or added.
            simulator_state = self.env.unwrapped.clone_state(include_rng=True)
            actions.append(action)
            traj_len += 1
            score += reward

            # Add cell if it is not in the archive. Update the archive if
            # current cell is better.
            # cell_representation = self.downsampler.process(state)
            cell_representation = self._downsample(state)
            if cell_representation not in self.archive:
                prev = Node(actions, prev=prev)
                cell_state = (simulator_state, prev, actions, traj_len, score)
                self.archive.add(cell_representation, cell_state)

                actions = []
                n_discoveries += 1
            else:
                cell = self.archive[cell_representation]
                if cell.should_update(score, traj_len):
                    prev = Node(actions, prev=prev)
                    cell_state = (simulator_state, prev,
                                  actions, traj_len, score)
                    self.archive.update(cell, cell_state)

                    actions = []
                    n_updates += 1

            # Increments visit count (and update cell weight internally).
            self.archive.increment_visits(cell_representation)

            # Track highscore, n_frames, n_updates, n_discoveries for logging.
            if score > self.highscore:
                self.highscore = score
            self.n_frames += 1
        return n_updates, n_discoveries


class Node:
    def __init__(self, actions=[], prev=None):
        self.actions = actions
        self.prev = prev
