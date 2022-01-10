from downsampler import CoarseBinarizer
from archive_selector import ReverseCountSelector
from goexplore import GoExplore
from agent import ActionRepetitionAgent
import argparse
parser = argparse.ArgumentParser(description='Run GoExplore phase 1.')

parser.add_argument('--game', type=str, default='Pong', help='Name of game')
args = parser.parse_args()


downsampler = CoarseBinarizer()
selector = ReverseCountSelector()
agent = ActionRepetitionAgent()
goexplore = GoExplore(agent, downsampler, selector, max_frames=2500000)

goexplore.run()
