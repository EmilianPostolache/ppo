import csv
import os
import signal

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
            self.writer.writeheader()
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    def display(self):
        keys = list(self.log_entry.keys())
        keys.sort()
        episode = self.log_entry['_episode']
        mean_return = self.log_entry['_mean_return']
        print(f'~~~~~~ Episode {episode}, Mean Return = {mean_return:.3f} ~~~~~')
        for key in keys:
            if key[0] != '_':
                print(f'{key}: {self.log_entry[key]:.5g}')
        print('\n')

    def log(self, items):
        self.log_entry.update(items)

    def close(self):
        self.file.close()


class GracefulExit:
    def __init__(self):
        self.exit = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signal, frame):
        print('You pressed Ctrl+C!')
        self.exit = True
