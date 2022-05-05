from collections import namedtuple
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

Data = namedtuple(
    'Data', 'scores n_cells n_updates n_discoveries iter_durations swarm')


class Logger:
    def __init__(self, params, folder=''):
        self.params = params
        self.folder = folder

    def add(self, names, values):
        self.logs = dict(zip(names, values))

    def save(self):
        # Create file name based on hyperparameters.
        filename = '_'.join(self.params[:-1])
        filename += f'_{self.params[-1]}'
        if self.folder:
            filename = f'{self.folder}/{filename}'

        # Extract trajectory and plotting data.
        trajectory = self.logs.pop('trajectory')
        scores = self.logs.pop('scores')
        n_cells = self.logs.pop('n_cells')
        n_updates = self.logs.pop('n_updates')
        n_discoveries = self.logs.pop('n_discoveries')
        iter_durations = self.logs.pop('iter_durations')

        # Save rest to json file.
        print(f'Saving results to "{filename}.json"')
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4)

        # Save trajectory.
        print(f'Saving trajectory to "{filename}.trajectory"')
        with open(filename + '.trajectory', 'wb') as f:
            pickle.dump(trajectory, f)

        # Save plotting data, and plot plotting data.
        data = Data(scores, n_cells, n_updates, n_discoveries, iter_durations)
        print(f'Saving plotting data to "{filename}.data"')
        with open(filename + '.data', 'wb') as f:
            pickle.dump(data, f)

        fig = self.plot_data(data)
        fig.savefig(f'{filename}.svg')

    def plot_data(self, data):
        """ Plot data. """
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(15, 10)

        gs = GridSpec(2, 3, figure=fig)

        ax_ncells = fig.add_subplot(gs[0, 0])
        ax_score = fig.add_subplot(gs[0, 1])
        ax_iterdur = fig.add_subplot(gs[0, 2])
        ax_celldisc = fig.add_subplot(gs[1, 0])
        ax_cellupd = fig.add_subplot(gs[1, 1])
        ax_total = fig.add_subplot(gs[1, 2])

        ax_ncells.plot(data.n_cells)
        ax_ncells.set_title('Number of cells')

        ax_score.plot(data.scores)
        ax_score.set_title('Highscore')

        N = 50
        K = int(len(data.iter_durations) / 5000)
        if K == 0:
            K = 1
        ax_iterdur.plot(np.convolve(
            data.iter_durations[::K], np.ones(N)/N, mode='valid'))
        ax_iterdur.set_title('Iteration duration (s)')

        total = np.array(data.n_discoveries) + np.array(data.n_updates)
        max_count = np.max(total)
        max_count += max_count*0.1
        ax_celldisc.plot(data.n_discoveries)
        ax_celldisc.set_title('Cell discoveries')
        ax_celldisc.set_ylim([0, max_count])

        ax_cellupd.plot(data.n_updates)
        ax_cellupd.set_title('Cell updates')
        ax_cellupd.set_ylim([0, max_count])

        ax_total.plot(np.convolve(
            total[::K], np.ones(N)/N, mode='valid'))
        ax_total.set_title('Cell discoveries + updates')

        ax_celldisc.set(ylabel='Count')
        return fig
