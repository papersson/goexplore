import argparse
import datetime
from pathlib import Path
from utils.logger import Logger
from algorithm.components.downsampler import CoarseBinarizer, UberReducer
from algorithm.components.archive_selector import RouletteWheel, StochasticAcceptance, Uniform
from algorithm.components.agent import ActionRepetitionAgent, RandomAgent
from algorithm.goexplore import GoExplore
import gym


def run_experiments(experiment_name, games, seeds, frames_grid, no_logger, agents, selectors, widths, heights, depths):
    # Create folder with format {date_experimentname}
    if not no_logger:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        path = Path(f'experiments/{date}_{experiment_name}')
        path.mkdir(exist_ok=True)

    # downsampler = UberReducer(8, 8, 8)
    downsampler = UberReducer(11, 8, 16)
    # downsampler = CoarseBinarizer()
    selector = StochasticAcceptance()
    # selector = ReverseCountSelector()

    for frames in frames_grid:
        for game in games:
            for agent in agents:
                env = gym.make(f'{game}Deterministic-v4')
                agent = RandomAgent(
                    env.action_space) if agent == 'random' else ActionRepetitionAgent(env.action_space)
                for seed in seeds:
                    for selector in selectors:
                        if selector == 'random':
                            selector = Uniform()
                        elif selector == 'roulette':
                            selector = RouletteWheel()
                        else:
                            selector = StochasticAcceptance()
                        for w in widths:
                            for h in heights:
                                for d in depths:
                                    downsampler = UberReducer(w, h, d)

                                    params = [f'{game}Deterministic-v4', agent.__class__.__name__,
                                              selector.__class__.__name__, str(downsampler), seed]
                                    logger = Logger(folder=str(path),
                                                    params=params) if not no_logger else None
                                    GoExplore(agent, downsampler, selector, seed=seed,
                                              max_frames=frames, env=env, verbose=True, logger=logger).run()


parser = argparse.ArgumentParser(description='test')

parser.add_argument('--exp_name', type=str, default='',
                    help='Required experiment name')
parser.add_argument('--games', type=str, nargs='+', default=['Pong'])
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[0], help='Experiment seed')
parser.add_argument('--frames', type=int, nargs='+', default=[1000],
                    help='Training frames')
parser.add_argument('--no_logger', dest='no_logger', action='store_true')
parser.add_argument('--agents', type=str, nargs='+', default=['ActRep'])
parser.add_argument('--selectors', type=str,
                    nargs='+', default=['StochAccept'])
parser.add_argument('--widths', type=int, nargs='+', default=[11])
parser.add_argument('--heights', type=int, nargs='+', default=[8])
parser.add_argument('--depths', type=int, nargs='+', default=[16])

# parser.add_argument('--no_logger', default=False,
#                     type=lambda x: (str(x).lower() == 'true'))
# parser.add_argument('--no_logger', default=False,
#                     action=argparse.BooleanOptionalAction)
# parser.add_argument('--sticky', type=float, nargs='+',
#                     default=[0.0], help='Action stickyness parameter')
# parser.add_argument('--max_steps', type=int, nargs='+',
#                     default=[100], help='Max steps during single exploration')
# parser.add_argument('--threshold', type=int, default=500000,
#                     help='Threshold criteria for termination')

args = parser.parse_args()

if __name__ == '__main__':
    run_experiments(args.exp_name, args.games, args.seeds,
                    args.frames, args.no_logger, args.agents, args.selectors, args.widths, args.heights, args.depths)
