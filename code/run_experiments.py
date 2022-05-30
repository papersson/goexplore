import argparse
import datetime
from pathlib import Path
from algorithm.logger import Logger
from algorithm.archive import RouletteWheel, StochasticAcceptance, Uniform
from algorithm.agent import ActionRepetitionAgent, RandomAgent
from algorithm.goexplore import GoExplore
import gym
from collections import namedtuple

CellParams = namedtuple('CellParams', 'width height depth')


def run_experiments(experiment_name, games, seeds, frames_grid, no_logging, agents, selectors, widths, heights, depths, max_cells):
    """ Run series of experiments.

    Example
    -------
    Run 20000 frames on Pong using stochastic acceptance selection:
    python run_experiments.py --games Pong --frames 20000 --selectors StochasticAcceptance

    """

    Path('experiments').mkdir(parents=True, exist_ok=True)
    if not no_logging:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        path = Path(f'experiments/{date}_{experiment_name}')
        path.mkdir(exist_ok=True)

    for frames in frames_grid:
        for game in games:
            for agent in agents:
                for seed in seeds:
                    for selector in selectors:
                        for w in widths:
                            for h in heights:
                                for d in depths:
                                    cell_params = CellParams(w, h, d)
                                    env = gym.make(f'{game}Deterministic-v4')
                                    agent = RandomAgent(
                                        env.action_space) if agent == 'Random' else ActionRepetitionAgent(env.action_space)

                                    if selector == 'Random':
                                        archive = Uniform(max_cells)
                                    elif selector == 'Roulette':
                                        archive = RouletteWheel(max_cells)
                                    else:
                                        archive = StochasticAcceptance(
                                            max_cells)

                                    params = [f'Frames{frames}', f'{game}Deterministic-v4', agent.__class__.__name__,
                                              archive.__class__.__name__, str(cell_params), f'Seed{seed}']
                                    logger = Logger(folder=str(path),
                                                    params=params) if not no_logging else None
                                    prettyparams = f'Experiment(frames={frames}, game={game}, agent={agent.__class__.__name__}, selector={archive.__class__.__name__}, downsampler={str(cell_params)}, seed={seed})'
                                    print(
                                        f"Starting experiment: {prettyparams}")
                                    goexplore = GoExplore(agent, cell_params, archive, seed=seed,
                                                          max_frames=frames, env=env, logger=logger)
                                    goexplore.run()


parser = argparse.ArgumentParser(description='test')

parser.add_argument('--exp-name', type=str, default='',
                    help='Experiment name')
parser.add_argument('--max-cells', type=int, nargs='+',
                    default=[100000], help='Max number of cells allowed due to memory constraints.')
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
                    args.frames, args.no_logging, args.agents, args.selectors, args.widths, args.heights, args.depths, args.max_cells[0])
