import argparse
import datetime
from pathlib import Path
from utils.logger import Logger
from algorithm.components.downsampler import CoarseBinarizer, UberReducer
from algorithm.components.archive_selector import RouletteWheel, StochasticAcceptance, Uniform
from algorithm.components.agent import ActionRepetitionAgent, RandomAgent
from algorithm.goexplore import GoExplore
import gym


def run_experiments(experiment_name, games, seeds, frames_grid, no_logging, agents, selectors, widths, heights, depths):
    # Create folder with format {date_experimentname} if we want to save logs/results
    if not no_logging:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        path = Path(f'experiments/{date}_{experiment_name}')
        path.mkdir(exist_ok=True)

    for frames in frames_grid:
        for game in games:
            for agent in agents:
                env = gym.make(f'{game}Deterministic-v4')
                agent = RandomAgent(
                    env.action_space) if agent == 'Random' else ActionRepetitionAgent(env.action_space)
                for seed in seeds:
                    for selector in selectors:
                        if selector == 'Random':
                            selector = Uniform()
                        elif selector == 'Roulette':
                            selector = RouletteWheel()
                        else:
                            selector = StochasticAcceptance()
                        for w in widths:
                            for h in heights:
                                for d in depths:
                                    downsampler = UberReducer(w, h, d)

                                    params = [f'Frames{frames}', f'{game}Deterministic-v4', agent.__class__.__name__,
                                              selector.__class__.__name__, str(downsampler), f'Seed{seed}']
                                    logger = Logger(folder=str(path),
                                                    params=params) if not no_logging else None
                                    GoExplore(agent, downsampler, selector, seed=seed,
                                              max_frames=frames, env=env, verbose=True, logger=logger).run()


parser = argparse.ArgumentParser(description='test')

parser.add_argument('--exp-name', type=str, default='',
                    help='Experiment name')
parser.add_argument('--games', type=str, nargs='+', default=['Pong'])
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[0], help='Experiment seed')
parser.add_argument('--frames', type=int, nargs='+', default=[1000],
                    help='Number of training frames')
parser.add_argument('--no-logging', dest='no_logging',
                    action='store_true', help='Do not log data during run.')
parser.add_argument('--agents', type=str, nargs='+',
                    default=['ActRep'], help='Agent: {ActionRepetition, Random}')
parser.add_argument('--selectors', type=str,
                    nargs='+', default=['StochAccept'], help='Cell selector: {Roulette, StochasticAcceptance, Random}')
parser.add_argument('--widths', type=int, nargs='+',
                    default=[11], help='Select downsampling width (default=11)')
parser.add_argument('--heights', type=int, nargs='+',
                    default=[8], help='Downsampling height (default=8)')
parser.add_argument('--depths', type=int, nargs='+',
                    default=[16], help='Downsampling pixel depth (default 16)')

args = parser.parse_args()

if __name__ == '__main__':
    run_experiments(args.exp_name, args.games, args.seeds,
                    args.frames, args.no_logging, args.agents, args.selectors, args.widths, args.heights, args.depths)
