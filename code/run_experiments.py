import argparse
import datetime
from pathlib import Path
from utils.logger import Logger
from algorithm.components.downsampler import CoarseBinarizer, UberReducer
from algorithm.components.archive_selector import ReverseCountSelector, UberSelector
from algorithm.components.agent import ActionRepetitionAgent
from algorithm.goexplore import GoExplore
import gym


def run_experiments(experiment_name, game, seeds, frames):
    # Create folder with format {date_experimentname}
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    path = Path(f'experiments/{date}_{experiment_name}')
    path.mkdir(exist_ok=True)

    downsampler = UberReducer(11, 8, 8)
    # downsampler = CoarseBinarizer()
    # selector = UberSelector()
    selector = ReverseCountSelector()
    env = gym.make(f'{game}Deterministic-v4')
    agent = ActionRepetitionAgent(env.action_space)

    for seed in seeds:
        params = [seed,
                  downsampler.__class__.__name__,
                  selector.__class__.__name__,
                  agent.__class__.__name__]
        logger = Logger(folder=str(path), params=params)
        GoExplore(agent, downsampler, selector, seed=seed,
                  max_frames=frames, env=env, verbose=True, logger=logger).run()


parser = argparse.ArgumentParser(description='test')

parser.add_argument('--exp_name', type=str, default='',
                    help='Required experiment name')
parser.add_argument('--game', type=str, default='Pong')
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[0], help='Experiment seed')
parser.add_argument('--frames', type=int, default=100000,
                    help='Training frames')
# parser.add_argument('--sticky', type=float, nargs='+',
#                     default=[0.0], help='Action stickyness parameter')
# parser.add_argument('--max_steps', type=int, nargs='+',
#                     default=[100], help='Max steps during single exploration')
# parser.add_argument('--threshold', type=int, default=500000,
#                     help='Threshold criteria for termination')

args = parser.parse_args()

if __name__ == '__main__':
    run_experiments(args.exp_name, args.game, args.seeds, args.frames)
