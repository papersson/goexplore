import argparse
import datetime
from pathlib import Path
from utils.logger import Logger
from algorithm.components.downsampler import CoarseBinarizer, UberReducer
from algorithm.components.archive_selector import ReverseCountSelector, UberSelector
from algorithm.components.agent import ActionRepetitionAgent
from algorithm.goexplore import GoExplore


def run_experiments(experiment_name, seeds):
    # Create folder with format {date_experimentname}
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    path = Path(f'experiments/{date}_{experiment_name}')
    path.mkdir(exist_ok=True)

    downsampler = UberReducer(11, 10, 8)
    selector = UberSelector()
    agent = ActionRepetitionAgent()

    for seed in seeds:
        params = [seed,
                  downsampler.__class__.__name__,
                  selector.__class__.__name__,
                  agent.__class__.__name__]
        logger = Logger(folder=str(path), params=params)
        GoExplore(agent, downsampler, selector, seed=seed,
                  max_frames=2000, game='Pong', logger=logger).run()


parser = argparse.ArgumentParser(description='test')

parser.add_argument('--exp_name', type=str, default='',
                    help='Required experiment name')
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[0], help='Experiment seed')
# parser.add_argument('--sticky', type=float, nargs='+',
#                     default=[0.0], help='Action stickyness parameter')
# parser.add_argument('--max_steps', type=int, nargs='+',
#                     default=[100], help='Max steps during single exploration')
# parser.add_argument('--threshold', type=int, default=500000,
#                     help='Threshold criteria for termination')

args = parser.parse_args()

if __name__ == '__main__':
    run_experiments(args.exp_name, args.seeds)
