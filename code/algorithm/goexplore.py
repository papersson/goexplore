from tqdm import trange
import numpy as np
import random
import time
from datetime import timedelta


MAX_FRAMES_PER_ITERATION = 100


class GoExplore:
    def __init__(self, agent, downsampler, archive,
                 env, seed, max_frames, verbose, logger):
        # Set seed, environment/game, agent, downsampler, and archive selector
        self.seed = seed
        self.env = env
        self.agent = agent
        self.downsampler = downsampler
        self.archive = archive
        self.max_frames = max_frames
        self.verbose = verbose
        self.logger = logger
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        self.action_graph = ActionGraph()

        # Track during run for plotting
        self.highscore, self.n_frames = 0, 0

    def run(self):
        start = time.time()

        # Add first cell to archive
        starting_state = self.env.reset()
        simulator_state = self.env.unwrapped.clone_state(include_rng=True)
        cell_representation = self.downsampler.process(starting_state)
        self.archive.initialize(cell_representation, simulator_state)

        n_iterations = int(self.max_frames / MAX_FRAMES_PER_ITERATION)
        scores, n_cells, n_updates_data, n_discoveries_data, iter_durations = [
        ], [], np.zeros(n_iterations), np.zeros(n_iterations), []
        dd = []
        with trange(n_iterations) as t:
            for i in t:
                iter_start = time.time()
                # Progress bar
                total = n_updates_data + n_discoveries_data
                mean_total = total.mean() + 0.0000001
                mean_recent = total[-10:].mean()
                saturation = (mean_recent - mean_total) / mean_total
                t.set_description(f'Iteration {i}')
                t.set_postfix(highscore=self.highscore, num_cells=len(self.archive), satur=saturation,
                              frames=(i+1) * MAX_FRAMES_PER_ITERATION)

                # Sample cell from archive
                cell = self.archive.sample()
                # self.archive.update_weight(cell)

                # cell.increment_visits()
                n_updates, n_discoveries = self._explore_from(cell)

                # Track for plotting
                iter_end = time.time()
                scores.append(self.highscore)
                n_cells.append(len(self.archive))
                n_updates_data[i] = n_updates
                n_discoveries_data[i] = n_discoveries
                iter_durations.append(round(iter_end - iter_start, 3))

                dd.append([cell.traj_len for cell in self.archive.cells])
        with open('mediumswarm.viz', 'wb') as f:
            import pickle
            pickle.dump(dd, f)

        # Extract cell that reached terminal state with highest score and smallest trajectory
        best_cell = self.archive.get_best_cell()
        print(best_cell)
        # traj = self.action_graph.get_trajectory(best_cell.latest_action)
        traj = best_cell.get_trajectory()
        swarm = [cell.traj_len for cell in self.archive.cells]

        # Save logs
        duration = (time.time() - start)
        if self.logger:
            names = ['highscore', 'traj_len', 'total_cells', 'duration', 'n_frames',
                     'trajectory', 'scores', 'n_cells', 'n_updates', 'n_discoveries', 'iter_durations', 'swarm']
            values = [self.highscore, len(traj), len(self.archive), str(timedelta(seconds=duration)),
                      self.n_frames, traj, scores, n_cells, n_updates_data, n_discoveries_data, iter_durations, swarm]
            self.logger.add(names, values)
            self.logger.save()

    def _explore_from(self, cell):
        n_updates, n_discoveries = 0, 0
        # Restore to cell's simulator state and retrieve its score, trajectory, and latest action
        traj_len, score = cell.load(self.env)

        # Maintain seen set to only increment visits at most once per exploration run
        cells_seen_during_iteration = set()
        n_steps = 0
        actions = []
        prev = cell.prev
        while (n_steps < MAX_FRAMES_PER_ITERATION):
            # Interact
            action = self.agent.act()
            state, reward, is_terminal, _ = self.env.step(action)

            # Track cell object state in case cell needs to be updated or added
            simulator_state = self.env.unwrapped.clone_state(
                include_rng=True)
            actions.append(action)
            # latest_action = self.action_graph.get(action, latest_action)
            traj_len += 1
            score += reward

            # node = Node()

            # Handle cell event. Cases:
            # Cell discovered: add to archive
            # Cell is better than archived cell: update cell in archive
            # Cell is not discovered or better: do nothing
            cell_representation = self.downsampler.process(state)
            if cell_representation not in self.archive:
                n_discoveries += 1
                # prev = node.prev if node else None
                prev = Node(actions, prev=prev)
                cell_state = (simulator_state, prev, actions, traj_len, score)
                self.archive.add(cell_representation, cell_state)
                actions = []
            else:
                cell = self.archive[cell_representation]
                if cell.should_update(score, traj_len):
                    n_updates += 1
                    prev = Node(actions, prev=prev)
                    cell_state = (simulator_state, prev,
                                  actions, traj_len, score)
                    self.archive.update(cell, cell_state)
                    actions = []
            self.archive.increment_visits(cell_representation)

            # Increment visit count/update weights if cell not seen during the episode
            # if cell_representation not in cells_seen_during_iteration:
            #     cells_seen_during_iteration.add(cell_representation)
            #     self.archive.update_weight(cell_representation)

            # Logging and termination check
            if score > self.highscore:
                self.highscore = score
            n_steps += 1
            self.n_frames += 1

            if is_terminal:
                cell.set_done()
        return n_updates, n_discoveries


class ActionGraph:
    def __init__(self):
        self.cache = {}

    def get(self, action, prev):
        if (action, prev) in self.cache:
            return self.cache[(action, prev)]
        else:
            latest_action = ActionNode(action, prev)
            self.cache[(action, prev)] = latest_action
            return latest_action

    def get_trajectory(self, node):
        actions = []
        while node:
            actions = [node.action] + actions
            node = node.prev
        return actions


class ActionNode:
    def __init__(self, action, prev=None):
        self.action = action
        self.prev = prev

    def __repr__(self):
        return str(self.action)


class Node:
    def __init__(self, actions=[], prev=None):
        self.actions = actions
        self.prev = prev
