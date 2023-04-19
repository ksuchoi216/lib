#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Project: [model]
# File: @/base.py
# Created Date: 2023-03-14 04:24:09
# Author: JAEHYUN YOO (jaehyun@algorigo.com)
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Last Modified:
# Modified By:
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Copyright (c) 2023 Algorigo Inc.
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
###

import h5py
from tensorflow.python.keras.saving import hdf5_format
import math
import time
import datetime
import sys
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

import os
# Hide Warnings for Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import pandas as pd
# import glob
# import numpy as np
# import random  # as rn
# from attrdict import AttrDict

# import tensorflow as tf
# from keras.models import Model, Sequential, load_model
# from keras.layers import Input, Dense, LSTM
# from keras.layers.core import Dense, Activation, Dropout
# from sklearn.preprocessing import MinMaxScaler

# import algolib
# from algolib.configuration import config, DBConfig
# from algolib.log import logger, func_log
# from algolib.error import AlgoError
# from forecast.constant import *


def str_to_datetime(value):
    if isinstance(value, str):
        if len(value) == 16:
            return datetime.datetime.strptime(value, '%Y-%m-%d %H:%M')
        elif len(value) == 19:
            return datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        elif len(value) > 19:
            return datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
    raise ValueError('not supported format for datetime:', value)


def to_datetime(value):
    # includes datetime.datetime, pd.Timestamp, ...
    if isinstance(value, datetime.datetime):
        return value
    elif isinstance(value, str):
        return str_to_datetime(value)
    elif isinstance(value, float) or isinstance(value, int):
        # https://stackoverflow.com/questions/2189800/how-to-find-length-of-digits-in-an-integer
        __length = int(math.log10(value)) + 1
        if __length == 10:
            return datetime.datetime.fromtimestamp(value)
        elif __length == 13:
            return datetime.datetime.fromtimestamp(value / 1000)
        else:
            raise ValueError(
                'invalid value for timestamp (10 digits for sec, 13 digits for ms)')
    raise ValueError('not supported format for datetime:', value)


def to_timestamp(value, unit='sec'):
    if isinstance(value, str):
        try:
            value = float(value)
        except:
            pass
        try:
            value = str_to_datetime(value)
        except:
            pass
    if isinstance(value, datetime.datetime):
        return time.mktime(value.timetuple())
    elif isinstance(value, int) or isinstance(value, float):
        __length = int(math.log10(value)) + 1
        if unit == 'sec' and __length == 10:
            return value
        elif unit == 'ms' and __length == 13:
            return value
        else:
            raise ValueError(
                'invalid timestamp (digits is required to be 10 or 13)')
    else:
        raise ValueError('not supported format for timestamp:', value)

# Example
#
# print(to_datetime(1671000201.984288))
# print(to_datetime(1671000201.984288*1000))
# print(to_datetime('2022-11-11 11:00:00'))
# print(to_datetime( pd.Timestamp(1513393355, unit='s', tz='US/Pacific') ))
#
# ts = time.time()
# dt = to_datetime(ts)
#
# print(ts, dt)
# print('-'*40)
# print(to_timestamp(ts))
# print(to_timestamp(ts*1000, unit='ms'))
# print(to_timestamp(dt))
# print(to_timestamp(str(dt), dt_format='%Y-%m-%d %H:%M:%S.%f'))

#
# Decorators
#


def not_implemented_method(func):
    """https://stackoverflow.com/questions/1151212/equivalent-of-notimplementederror-for-fields-in-python
    """
    from functools import wraps
    from inspect import getargspec, formatargspec

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        c = self.__class__.__name__
        m = func.__name__
        a = formatargspec(*getargspec(func))
        raise NotImplementedError(
            '\'%s\' object does not implement the method \'%s%s\'' % (c, m, a))
    return wrapper


def save_model_h5(model, model_path, attributes):
    with h5py.File(model_path, mode='w') as f:
        hdf5_format.save_model_to_hdf5(model, f)
        f.attrs.update(attributes)


def load_model_h5(model_path):
    with h5py.File(model_path, mode='r') as f:
        __model = hdf5_format.load_model_from_hdf5(f)
        __meta = dict(f.attrs)
    return __model, __meta

# version = __meta.get('version', None)
# coeff_minmax = __meta.get('coeff_minmax', None)


# def save_version(self, version):
#     if not isinstance(version, str):
#         version = str(version)
#     file = None
#     try:
#         file = open(f'./model/{self.name}.version', 'w')
#         file.write(version)
#     except e as Exception:
#         print(e)
#     finally:
#         if file is not None:
#             file.close()

# def load_version(self):
#     version = None
#     file = None
#     file_path = f'./model/{self.name}.version'
#     if os.path.isfile(file_path):
#         try:
#             file = open(file_path)
#             version = file.read()
#         except e as Exception:
#             print(e)
#         finally:
#             if file is not None:
#                 file.close()
#     try:
#         return to_timestamp(version)
#     except e as Exception:
#         print(e)
#     finally:
#         return None


class ModelBase:
    # @not_implemented_method
    # def load_model(self, file_prefix=None):
    #     pass

    # @staticmethod
    def is_valid(cls, pred_time=None):
        return True

    @not_implemented_method
    def prepare(self, options):
        pass

    @not_implemented_method
    def fit(self):
        pass

    @not_implemented_method
    def predict(self):
        pass
