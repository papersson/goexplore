from collections import namedtuple
import datetime
import json
import pickle

Data = namedtuple('Data', 'scores n_cells iter_durations')


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
        iter_durations = self.logs.pop('iter_durations')
        print(f'Saving results to "{filename}.json"')
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4)

        print(f'Saving trajectory to "{filename}.trajectory"')
        with open(filename + '.trajectory', 'wb') as f:
            pickle.dump(trajectory, f)

        data = Data(scores, n_cells, iter_durations)
        print(f'Saving plotting data to "{filename}.data"')
        with open(filename + '.data', 'wb') as f:
            pickle.dump(data, f)
