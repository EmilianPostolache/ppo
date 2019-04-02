import csv
import os

LOG_DIR = 'logs'


class Logger:
    def __init__(self, name, timestamp):
        log_dir = os.path.join(LOG_DIR, name, timestamp)
        os.makedirs(log_dir)
        path = os.path.join(log_dir, 'log.csv')
        self.file = open(path, 'w')
        self.log_entry = {}
        self.writer = None

    def write(self, display=True):
        if display:
            self.display()
        if not self.writer:
            fieldnames = list(self.log_entry.keys())
            fieldnames.sort()
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    def display(self):
        keys = list(self.log_entry.keys())
        keys.sort()
        episode = self.log_entry['_episode']
        mean_reward = self.log_entry['_mean_reward']
        print(f'~~~~~~ Episode {episode}, Mean Return = {mean_reward:1.f} ~~~~~')
        for key in keys:
            if key[0] != '_':
                print(f'{key}: {self.log_entry[key]:.5g}')
        print('\n')

    def log(self, items):
        self.log_entry.update(items)

    def close(self):
        self.file.close()
