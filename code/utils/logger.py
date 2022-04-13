from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle

Data = namedtuple(
    'Data', 'scores n_cells n_updates n_discoveries iter_durations')


class Logger:
    def __init__(self, params, folder=''):
        self.params = params
        self.folder = folder

    def add(self, names, values):
        self.logs = dict(zip(names, values))

    def save(self, experiment_name=''):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        filename = '_'.join(self.params[:-1])
        filename += f'_Seed{self.params[-1]}'
        if self.folder:
            filename = f'{self.folder}/{filename}'

        trajectory = self.logs.pop('trajectory')
        scores = self.logs.pop('scores')
        n_cells = self.logs.pop('n_cells')
        n_updates = self.logs.pop('n_updates')
        n_discoveries = self.logs.pop('n_discoveries')
        iter_durations = self.logs.pop('iter_durations')
        print(f'Saving results to "{filename}.json"')
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4)

        print(f'Saving trajectory to "{filename}.trajectory"')
        with open(filename + '.trajectory', 'wb') as f:
            pickle.dump(trajectory, f)

        data = Data(scores, n_cells, n_updates, n_discoveries, iter_durations)
        print(f'Saving plotting data to "{filename}.data"')
        with open(filename + '.data', 'wb') as f:
            pickle.dump(data, f)

        total = np.array(n_updates) + np.array(n_discoveries)
        max_count = np.max(total)
        max_count += max_count * 0.1

        N = 50
        K = int(len(iter_durations) / 5000)
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(15, 10)

        axs[0, 0].plot(n_cells)
        axs[0, 0].set_title('Number of cells')

        axs[0, 1].plot(scores)
        axs[0, 1].set_title('Highscore')

        N = 50
        K = int(len(iter_durations) / 5000)
        axs[0, 2].plot(np.convolve(iter_durations[::K],
                       np.ones(N)/N, mode='valid'))
        axs[0, 2].set_title('Iteration duration (s)')

        axs[1, 0].plot(n_discoveries)
        axs[1, 0].set_title('Cell discoveries')
        axs[1, 0].set_ylim([0, max_count])

        axs[1, 1].plot(n_updates)
        axs[1, 1].set_title('Cell updates')
        axs[1, 1].set_ylim([0, max_count])

        axs[1, 2].plot(total)
        axs[1, 2].set_title('Cell discoveries + updates')
        axs[1, 2].set_ylim([0, max_count])

        axs[1, 0].set(ylabel='Count')

        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.svg')
        fig.savefig(f'{filename}.eps')
