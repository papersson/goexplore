from collections import namedtuple
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle
import seaborn as sns

Data = namedtuple(
    'Data', 'scores n_cells n_updates n_discoveries iter_durations swarm')


class Logger:
    def __init__(self, params, folder=''):
        self.params = params
        self.folder = folder

    def add(self, names, values):
        self.logs = dict(zip(names, values))

    def save(self, experiment_name=''):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        filename = '_'.join(self.params[:-1])
        filename += f'_{self.params[-1]}'
        if self.folder:
            filename = f'{self.folder}/{filename}'

        trajectory = self.logs.pop('trajectory')
        scores = self.logs.pop('scores')
        n_cells = self.logs.pop('n_cells')
        n_updates = self.logs.pop('n_updates')
        n_discoveries = self.logs.pop('n_discoveries')
        iter_durations = self.logs.pop('iter_durations')
        swarm = self.logs.pop('swarm')
        print(f'Saving results to "{filename}.json"')
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4)

        print(f'Saving trajectory to "{filename}.trajectory"')
        with open(filename + '.trajectory', 'wb') as f:
            pickle.dump(trajectory, f)

        data = Data(scores, n_cells, n_updates,
                    n_discoveries, iter_durations, swarm)
        print(f'Saving plotting data to "{filename}.data"')
        with open(filename + '.data', 'wb') as f:
            pickle.dump(data, f)

        # total = np.array(n_updates) + np.array(n_discoveries)
        # max_count = np.max(total)
        # max_count += max_count * 0.1

        # N = 50
        # K = int(len(iter_durations) / 5000)
        # if K == 0:
        #     K = 1
        # fig, axs = plt.subplots(2, 3)
        # fig.set_size_inches(15, 10)

        # axs[0, 0].plot(n_cells)
        # axs[0, 0].set_title('Number of cells')

        # axs[0, 1].plot(scores)
        # axs[0, 1].set_title('Highscore')

        # axs[0, 2].plot(np.convolve(iter_durations[::K],
        #                np.ones(N)/N, mode='valid'))
        # axs[0, 2].set_title('Iteration duration (s)')

        # axs[1, 0].plot(n_discoveries)
        # axs[1, 0].set_title('Cell discoveries')
        # axs[1, 0].set_ylim([0, max_count])

        # axs[1, 1].plot(n_updates)
        # axs[1, 1].set_title('Cell updates')
        # axs[1, 1].set_ylim([0, max_count])

        # axs[1, 2].plot(total)
        # axs[1, 2].set_title('Cell discoveries + updates')
        # axs[1, 2].set_ylim([0, max_count])

        # axs[1, 0].set(ylabel='Count')

        # fig.savefig(f'{filename}.png')
        fig = self.plot_data(data)
        fig.savefig(f'{filename}.svg')
        # fig.savefig(f'{filename}.eps')

    def plot_data(self, data):
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(15, 10)

        gs = GridSpec(3, 3, figure=fig)

        ax_ncells = fig.add_subplot(gs[0, 0])
        ax_score = fig.add_subplot(gs[0, 1])
        ax_iterdur = fig.add_subplot(gs[0, 2])
        ax_celldisc = fig.add_subplot(gs[1, 0])
        ax_cellupd = fig.add_subplot(gs[1, 1])
        ax_total = fig.add_subplot(gs[1, 2])
        ax_swarm = fig.add_subplot(gs[2, :])

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
        # ax_total.plt(total)
        ax_total.set_title('Cell discoveries + updates')

        ax_celldisc.set(ylabel='Count')

        # ax = plt.subplot(3, 1, 1)
        if data.n_cells[-1] < 5000:
            ax_swarm = self.plot_swarm(data.swarm)
        return fig

    def plot_swarm(self, swarm):
        g = sns.swarmplot(y=swarm, orient='v', size=1)
        sns.despine(bottom=True, left=True)
        g.set(ylabel=None)
        g.tick_params(left=False)
        return g
