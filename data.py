import os
import random
import torch
import numpy as np
import pandas as pd


class BattDataset(object):

    def __init__(self, x, y, length=None, size=10, shuffle=True, torch_ready=False, random_seed=None):
        self.x = None
        self.y = None
        self.length = length[0]
        self.shuffle = shuffle
        self.size = size
        if random_seed is not None:
            random.seed(random_seed)
        self.__build_dataset(x, y, torch_ready)

    def __getitem__(self, x, y, index):
        idx = random.randint(0, x.shape[1] - self.length)
        xi = np.zeros(x.shape[1])
        xi[idx:(idx+self.length)] = x[index, idx:(idx+self.length)]
        yi = y[index]
        return xi, yi

    def __build_dataset(self, x, y, torch_ready=False):
        x_data = np.zeros((self.size * x.shape[0], x.shape[1]))
        y_data = np.zeros(self.size * x.shape[0])
        i = 0
        for n in range(self.size):
            for idx in range(x.shape[0]):
                x_data[i], y_data[i] = self.__getitem__(x, y, idx)
                i += 1
        if self.shuffle:
            idx = np.arange(x_data.shape[0])
            random.shuffle(idx)
            x_data = x_data[idx]
            y_data = y_data[idx]

        if torch_ready:
            self.x = torch.tensor(x_data).float()
            self.y = torch.tensor(y_data[:, None]).float()
        else:
            self.x = x_data
            self.y = y_data


def load_dataset(dataset):
    path = os.path.join('.', 'data', f'Dataset_Step{dataset}mV_dis_all_pos.csv')
    data = pd.read_csv(path, header=None).to_numpy()
    y = data[:, -1]
    x = data[:, :-1]
    x_mean = np.mean(x[:, :-1])
    x_std = np.std(x[:, :-1])
    x[:, :-1] = (x[:, :-1] - x_mean) / x_std
    return x, y
