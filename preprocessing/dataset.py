import sys
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class Base_Dataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray):
        self.data_x = data_x
        self.data_y = data_y
        print(f'dataset shape: {data_x.shape} {data_y.shape}')
        print(f'dataset type: {type(data_x)} {type(data_y)}')

    def __len__(self):
        length, _ = self.data_x.shape
        return length

    def __getitem__(self, index):
        x = torch.tensor(self.data_x[index], dtype=torch.float32)
        x = torch.reshape(x, (-1,))
        y = torch.tensor(self.data_y[index], dtype=torch.float32)

        return x, y


class LSTM_Dataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, seq_len=5):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        print(f'dataset load shape: {data_x.shape} {data_y.shape}')
        print(f'dataset load type: {type(data_x)} {type(data_y)}')

    def __len__(self):
        length, _ = self.data_x.shape
        return length

    def __getitem__(self, index):
        start = index
        end = index + self.seq_len
        x = torch.tensor(self.data_x[start: end], dtype=torch.float32)
        y = torch.tensor(self.data_y[start: end], dtype=torch.float32)

        return x, y


# DATALOADER ============================================================


def make_dataloaders(dataset, ratio=[0.7, 0.2], batch_size=4) -> dict:
    num = len(dataset)
    train_num = int(num * ratio[0])
    val_num = int(num * ratio[1])
    test_num = num - train_num - val_num
    print(f'split data into [{train_num}, {val_num}, {test_num}]')

    splited_dataset = random_split(
        dataset, [train_num, val_num, test_num])

    datanames = ['train', 'val', 'test']
    dataloaders = {}
    for dataset, dataname in zip(splited_dataset, datanames):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        dataloaders[dataname] = dataloader

    return dataloaders
