import csv
import os
import signal
import numpy as np

LOG_DIR = 'logs'


class VecScaler:
    # Tricks: - strange division by 3 on std dev  !

    def __init__(self, obs_dim):
        self.var = np.zeros(obs_dim)
        self.mean = np.zeros(obs_dim)
        self.m = 0
        self.first_pass = True

    def update(self, x):
        if self.first_pass:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_mean = ((self.mean * self.m) + (new_data_mean * n)) / (self.m + n)
            self.var = ((self.m * (self.var + np.square(self.mean))) +
                        (n * (new_data_var + np.square(new_data_mean)))) / (self.m + n) - (np.square(new_mean))
            self.var = np.maximum(0.0, self.var)
            self.mean = new_mean
            self.m += n

    def get(self):
        # print('get:', 1/(np.sqrt(self.var) + 0.1) / 3)
        try:
            return 1/(np.sqrt(self.var) + 0.1) / 3, self.mean
        except RuntimeWarning:
            print(self.var)
            exit()


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
        print('~~~~~~ Episode {}, Mean Return = {:.3f} ~~~~~'.format(episode, mean_return))
        for key in keys:
            if key[0] != '_':
                print('{}: {:.5g}'.format(key, self.log_entry[key]))
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
