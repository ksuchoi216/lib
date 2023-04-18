from abc import ABC, abstractmethod
from typing import Tuple

import os
import numpy as np
import pandas as pd
import re

# import torch
# from torch.utils.data import Dataset, DataLoader, random_split

# PREPROCESS ============================================================


class processor(ABC):
    @abstractmethod
    def load_y(self):
        pass

    @abstractmethod
    def load_x(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class Preprocessor_resp_range(processor):
    def __init__(self, cfg, task='cls'):
        self.raw_data_dir = cfg['raw_data_dir']

        file_nums = []
        for id_num in cfg['id_nums']:
            for data_num in cfg['data_nums']:
                file_num = f'{id_num:02}{data_num:02}'
                file_nums.append(file_num)

        self.file_nums = file_nums

        path = cfg['save_data_dir']
        if not os.path.exists(path):
            os.mkdir(path)

        if task is 'cls':
            save_data_dir = f'{path}/cls'
        elif task is 'reg':
            save_data_dir = f'{path}/reg'
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)

        self.path_save_x = f'{save_data_dir}/data_x_resp.npy'
        self.path_save_y = f'{save_data_dir}/data_y_resp.npy'

    def load_y(self, file_num):
        path_res = f'{self.raw_data_dir}/res.csv'
        dfy = pd.read_csv(path_res)
        # print(f'y path: {path_res} y shape: {dfy.shape}')
        dfy = dfy[(dfy['range'] == 'partial') & (
            dfy['file_num'] == int(file_num))]

        dfy = dfy[['file_num', 'fsr_num', 'start', 'end',
                   'p_resp_peak', 'p_resp_freq', 'p_hr_freq']]

        dfy['l_resp_peak'] = dfy['p_resp_peak'].apply(
            lambda x: 1 if x > 90 else 0)
        dfy['l_resp_freq'] = dfy['p_resp_freq'].apply(
            lambda x: 1 if x > 90 else 0)
        # dfy['l_hr_freq'] = dfy['p_hr_freq'].apply(
        # lambda x: 1 if x > 70 else 0)
        # dfy.loc[((dfy['l_resp_peak'] == 1) | (
        #     dfy['l_resp_freq'] == 1)), 'label'] = 1
        dfy.loc[(dfy['l_resp_freq'] == 1), 'label'] = 1
        dfy.loc[dfy['label'] != 1, 'label'] = 0
        dfy['label'] = dfy['label'].astype(np.int)
        # display(dfy.head())

        self.fsr_nums = dfy['fsr_num'].to_numpy()
        self.starts = dfy['start'].to_numpy(np.int)
        self.ends = dfy['end'].to_numpy(np.int)
        data_y = dfy['label'].to_numpy(np.int)

        return data_y

    def load_x(self, file_num):
        path_fsr = f'{self.raw_data_dir}/fsr_{file_num}.csv'
        dfx = pd.read_csv(path_fsr)
        # print(f'x path: {path_fsr} x shape: {dfx.shape}')
        # display(dfx.head())
        # dfx = dfx[dfx['file_num'] == file_num]
        fsrs = []

        for i, row in dfx['fsr_org'].iteritems():
            row = list(re.findall('[0-9]+', row))
            row = [int(x) for x in row]
            fsrs.append(row)
        fsrs = np.vstack(fsrs)
        # print(f'fsrs shape: {fsrs.shape}')

        _fsrs = []
        for start, end, fsr_num in zip(self.starts, self.ends, self.fsr_nums):
            fsr = fsrs[start:end, fsr_num]
            _fsrs.append(fsr)

        data_x = np.array(_fsrs)

        return data_x

    @ staticmethod
    def MinMaxScale(array, min, max):
        return (array - min) / (max - min)

    def __call__(self):

        # if os.path.exists(path_data_x) and os.path.exists(path_data_y):
        #     all_data_x = np.load(path_data_x)
        #     all_data_y = np.load(path_data_y)

        all_data_x = []
        all_data_y = []
        for i, file_num in enumerate(self.file_nums):
            if i == 1:

            data_y = self.load_y(file_num)
            data_x = self.load_x(file_num)
            print(f'{file_num}: data_x shape: {data_x.shape}')
            all_data_x.append(data_x)
            all_data_y.append(data_y)

        all_data_x = np.concatenate(all_data_x, axis=0)
        all_data_y = np.concatenate(all_data_y)
        MIN, MAX = all_data_x.min(), all_data_x.max()
        # all_data_x = self.MinMaxScale(all_data_x, MIN, MAX)

        np.save(self.path_save_x, all_data_x)
        np.save(self.path_save_y, all_data_y)

        print(f'data shape: {all_data_x.shape} {all_data_y.shape}')
        return all_data_x, all_data_y


class Preprocessor_resp_reg(Preprocessor_resp_range):
    def __init__(self, cfg, task='reg'):
        super().__init__(cfg, task)

    def load_y(self, file_num):
        path_res = f'{self.raw_data_dir}/res.csv'
        dfy = pd.read_csv(path_res)
        print(f'y path: {path_res} y shape: {dfy.shape}')
        dfy = dfy[(dfy['range'] == 'partial') & (
            dfy['file_num'] == int(file_num))]
        dfy = dfy[['file_num', 'fsr_num', 'start', 'end', 'resp_ref_key']]
        # display(dfy)
        dfy['label'] = dfy['resp_ref_key']

        self.fsr_nums = dfy['fsr_num'].to_numpy()
        self.starts = dfy['start'].to_numpy(np.int)
        self.ends = dfy['end'].to_numpy(np.int)
        data_y = dfy['label'].to_numpy(np.int)

        return data_y


'''
class Preprocessor_all(processor):
    def __init__(self, raw_data_dir):
        self.data_dir = data_dir
        self.fsr_filenums = np.arange(1, 10)
        print(f'fsr_filenums: {self.fsr_filenums}')

    def load(self, file_num):
        # load y
        path_res = f'{self.data_dir}/res.csv'
        dfy = pd.read_csv(path_res)
        # print(f'y path: {path} y shape: {dfy.shape}')
        dfy = dfy[(dfy['range'] == 'partial') & (
            dfy['file_num'] == int(file_num))]

        dfy = dfy[['file_num', 'fsr_num', 'start', 'end',
                   'p_resp_peak', 'p_resp_freq', 'p_hr_freq']]

        dfy['l_resp_peak'] = dfy['p_resp_peak'].apply(
            lambda x: 1 if x > 70 else 0)
        dfy['l_resp_freq'] = dfy['p_resp_freq'].apply(
            lambda x: 1 if x > 70 else 0)
        dfy['l_hr_freq'] = dfy['p_hr_freq'].apply(lambda x: 1 if x > 70 else 0)
        dfy.loc[((dfy['l_resp_peak'] == 1) | (dfy['l_resp_freq'] == 1))
                & (dfy['l_hr_freq'] == 1), 'label'] = 1
        dfy.loc[dfy['label'] != 1, 'label'] = 0
        dfy['label'] = dfy['label'].astype(np.int)
        display(dfy.head())  # dfy

        fsr_nums = dfy['fsr_num'].to_numpy()
        starts = dfy['start'].to_numpy(np.int)
        ends = dfy['end'].to_numpy(np.int)
        data_y = dfy['label'].to_numpy(np.int)
        # data_y = [int(y) for y in ]

        # laod X
        path_fsr = f'{self.data_dir}/fsr_{file_num}.csv'
        dfx = pd.read_csv(path_fsr)
        # print(f'x path: {path} x shape: {dfx.shape}')
        display(dfx.head())
        # dfx = dfx[dfx['file_num'] == file_num]
        fsrs = []

        for i, row in dfx['fsr_org'].iteritems():
            row = list(re.findall('[0-9]+', row))
            row = [int(x) for x in row]
            fsrs.append(row)
        fsrs = np.vstack(fsrs)
        # print(f'fsrs shape: {fsrs.shape}')

        _fsrs = []
        for start, end, fsr_num in zip(starts, ends, fsr_nums):
            fsr = fsrs[start:end, fsr_num]
            _fsrs.append(fsr)

        data_x = np.array(_fsrs)
        print(f'{file_num}: data shape: {data_x.shape} {data_y.shape}')

        return data_x, data_y

    @staticmethod
    def MinMaxScale(array, min, max):
        return (array - min) / (max - min)

    def __call__(self):
        path_data_x = f'{self.data_dir}/data_x_all.npy'
        path_data_y = f'{self.data_dir}/data_y_all.npy'

        if os.path.exists(path_data_x) and os.path.exists(path_data_y):
            all_data_x = np.load(path_data_x)
            all_data_y = np.load(path_data_y)
        else:
            all_data_x = []
            all_data_y = []
            for filenum in self.fsr_filenums:
                filenum = f'10{filenum:02}'
                data_x, data_y = self.load(filenum)
                all_data_x.append(data_x)
                all_data_y.append(data_y)
            all_data_x = np.concatenate(all_data_x, axis=0)
            all_data_y = np.concatenate(all_data_y)

            MIN = all_data_x.min()
            MAX = all_data_x.max()

            all_data_x = self.MinMaxScale(all_data_x, MIN, MAX)

            np.save(path_data_x, all_data_x)
            np.save(path_data_y, all_data_y)

        print(
            f'all_data shape: {all_data_x.shape} {all_data_y.shape}')

        return all_data_x, all_data_y
'''
