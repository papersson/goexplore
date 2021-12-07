import random
import gym
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from copy import deepcopy
import datetime
from pathlib import Path
import time
from datetime import timedelta
import json
from collections import namedtuple

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return random.choice(self.action_space)

# From https://github.com/uber-research/go-explore/blob/240056852514ffc39f62d32ae7590a39fd1814b9/policy_based/goexplore_py/explorers.py#L26
# Repeats actions with 95% probability
# TODO: is it equivalent to sticky actions?
class ActionRepetitionAgent:
    def __init__(self, action_space, mean_repeat=20):
        self.action_space = action_space
        self.mean_repeat = mean_repeat
        self.action = 0 # noop
        self.remaining = 0

    def act(self):
        if self.remaining <= 0:
            self.remaining = np.random.geometric(1 / self.mean_repeat)
            self.action = random.choice(self.action_space)
        self.remaining -= 1
        return self.action

# https://gist.github.com/mttk/74dc6eaaea83b9f06c2cc99584d45f96
# Larger rendering video
from gym.envs.classic_control import rendering
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
viewer = rendering.SimpleImageViewer()

class Cell:
    def __init__(self, simulator_state, actions_taken, score):
        self.visits = 0
        self.done = False
        self.update(simulator_state, actions_taken, score)
        
    def update(self, simulator_state, actions_taken, score):
        self.simulator_state = simulator_state
        self.actions_taken = actions_taken
        self.score = score
    
    def increment_visits(self):
        self.visits += 1
        
    def restore_state(self, env):
        env.unwrapped.restore_state(self.simulator_state)
        
    def history(self):
        return deepcopy(self.actions_taken), deepcopy(self.score)
    
    def is_worse(self, score, actions_taken_length):
        return ((score > self.score) 
                or (score == self.score and actions_taken_length < len(self.actions_taken)))
    
    def set_done(self):
        self.done = True
    
    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={len(self.actions_taken)}, visits={self.visits}, done={self.done})'
    
    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length
    
    def __lt__(self, other):
        return (-self.score, len(self.actions_taken)) < (-other.score, len(other.actions_taken))
    
def cell_repr(img):
    # Crop and resize
    img = img[34:194:2, ::2]

    # Convert to greyscale
    img = img.mean(axis=2)

    # Shrink
    img = ndimage.interpolation.zoom(img, 0.1)
    
    # Binarize
    img = np.round(img, 2)
    threshold = 77.7
    img[img < threshold] = 0
    img[img >= threshold] = 1
    
    return tuple(img.flatten())

def explore(env, agent, archive, cell, stickyness, maxsteps, n_iterations, n_frames, highscore, visualize=False):
    state = env.reset()
    cell.restore_state(env)
    actions_taken, score = cell.history()

    done = False
    n_steps = 0
    seen_cells = set() # Track cells seen during the episode

    action = agent.act()
    while (not done and n_steps < maxsteps):
        # Interact
        if random.random() > stickyness:
            action = agent.act()

        state, reward, done, _ = env.step(action)
        actions_taken.append(action)

        score += reward
        if score > highscore:
            highscore = score
            print(f'New highscore: {highscore}')

        # Update or add cell to archive
        cell_representation = cell_repr(state)
        cell = _update_or_create_cell(archive, cell_representation, env, score, deepcopy(actions_taken))

        # Increment visit count if cell not seen during the episode
        if cell_representation not in seen_cells:
            seen_cells.add(cell_representation)
            cell.increment_visits()

        n_steps += 1
        n_frames += 1
        if done:
            cell.set_done()
            break

    n_iterations += 1
    return highscore, n_frames, n_iterations

def _update_or_create_cell(archive, cell_representation, env, score, actions_taken):
    if cell_representation in archive:
        cell = archive[cell_representation]
        if cell.is_worse(score, len(actions_taken)):
            simulator_state = env.unwrapped.clone_state(include_rng=True)
            cell.update(simulator_state, actions_taken, score)
    else:
        simulator_state = env.unwrapped.clone_state(include_rng=True)
        cell = Cell(simulator_state, actions_taken, score)
        archive[cell_representation] = cell
    return cell

Experience = namedtuple('Experience', 'state action reward done')

def run_experiment(agent, params, below_threshold, path):
    stickyness, max_steps, seed, *comment = params.values()
    print(f'\n\nStarting Experiment(seed={seed}, stickyness={stickyness}, max_steps={max_steps})')

    start = time.time()
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make('PongDeterministic-v4')
    env.seed(seed)
    env.action_space.seed(seed)

    # Initialize state
    start_state = env.reset()

    # Create initial cell
    simulator_state = env.unwrapped.clone_state(include_rng=True)
    actions_taken = []
    score = 0.0
    cell = Cell(simulator_state, actions_taken, score)
    cell.increment_visits()

    # Create archive and add initial cell
    archive = {}
    cell_representation = cell_repr(start_state)
    archive[cell_representation] = cell

    # Explore until step threshold reached
    highscore, n_frames, n_iterations = 0, 0, 0
    logs = {}
    scores = []
    n_cells = []
    for _ in tqdm(while_generator(below_threshold(n_frames))):
        if not below_threshold(n_frames):
            break

        visits = [cell.visits for cell in archive.values()]
        #weights = [1 / np.log(c.visits + 1) for c in archive.values()]
        #probs = [w / sum(weights) for w in weights]
        rev_counts = [max(visits) + 1 - v for v in visits]
        probs = [v / sum(rev_counts) for v in rev_counts]
        cell = np.random.choice(list(archive.values()), 1, p=probs)[0]

        highscore, n_frames, n_iterations = explore(env, agent, archive, cell, stickyness, max_steps, n_iterations, n_frames, highscore)

        scores.append(highscore)
        n_cells.append(len(archive))

        if n_frames % 500000 == 0:
            print(f'Iterations: {n_iterations}\tFrames: {n_frames}\tScore: {highscore}\t Cells: {len(archive)}')

    # Extract cell that reached terminal state with highest score and smallest trajectory
    cells = list(archive.values())
    solved_cells = [cell for cell in cells if cell.done is True]
    best_cell = sorted(solved_cells)[0] if solved_cells else sorted(cells)[0]

    # Save logs to json file
    elapsed = (time.time() - start)
    logs['time'] = str(timedelta(seconds=elapsed))
    logs['n_frames'] = n_frames
    logs['n_iterations'] = n_iterations
    logs['highscore'] = highscore
    logs['actions_taken'] = best_cell.actions_taken
    logs['scores'] = scores
    logs['n_cells'] = n_cells
    save(logs, path, params)

    print(f'Highscore: {highscore}')
    print('Experiment done.')
    return highscore

# https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop
def while_generator(condition):
    while condition:
        yield

def save(logs, path, params):
    file_name = ''
    for name, value in params.items():
        file_name += f'{name}{value}_'
    file_name += '.json'
    file_path = path / file_name
    with file_path.open('w', encoding='utf-8') as fp:
        json.dump(logs, fp, indent=4)
    print(f'Saving results to "{file_path}"')

def run_experiments(experiment_name, seeds, stickyness_grid, max_steps_grid, below_threshold):
    # Create folder with format {date_experimentname}
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    path = Path(f'experiments/{date}_{experiment_name}')
    path.mkdir(exist_ok=True)
    
    action_space = [0, 2, 3] # noop, up, down
    agent = ActionRepetitionAgent(action_space)
    comment = ''
    for seed in seeds:
        for stickyness in stickyness_grid:
            for max_steps in max_steps_grid:
                params = {'stickyness': stickyness, 'maxsteps': max_steps, 'seed': seed}
                run_experiment(agent, params, below_threshold, path)

import argparse
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--exp_name', type=str, help='Required experiment name')
parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Experiment seed')
parser.add_argument('--sticky', type=float, nargs='+', default=[0.0], help='Action stickyness parameter')
parser.add_argument('--max_steps', type=int, nargs='+', default=[100], help='Max steps during single exploration')
parser.add_argument('--term_threshold', type=int, default=500000, help='Threshold criteria for termination')

args = parser.parse_args()
below_threshold = lambda x: x < args.term_threshold

run_experiments(args.exp_name, args.seeds, args.sticky, args.max_steps, below_threshold)
