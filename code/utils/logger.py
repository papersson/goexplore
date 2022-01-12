import datetime
import json


class Logger:
    def __init__(self, params, folder=''):
        self.params = params
        self.folder = folder

    def add(self, names, values):
        self.logs = dict(zip(names, values))

    def save(self, experiment_name=''):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        filename = '_'.join(self.params[1:])
        filename += f'_Seed{self.params[0]}'
        if self.folder:
            filename = f'{self.folder}/{filename}.json'
        else:
            filename = f'experiments/{date}_seed{self.seed}_{experiment_name}.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4)
        print(f'Saving results to "{filename}"')
