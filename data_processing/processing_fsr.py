import os
import csv
import time
import copy
import datetime
import math

# import jsonlines

import pandas as pd
import numpy as np
from scipy import signal, interpolate

# from clib.utils import plot_simple, reorder_columns


def reorder_columns(df, reorder_col_names: list):
    col_names = copy.deepcopy(reorder_col_names)
    col_names.reverse()
    for col_name in col_names:
        # print(col_name)
        col = df.pop(col_name)
        df.insert(0, col.name, col)


#############################################################################
# base
#############################################################################


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

#############################################################################
# get data at the initial stage
#############################################################################


def get_path_v1(source_path, folder_name, file_num):
    fsr_file_name = f'fsr_{file_num}.csv'
    ppg_file_name = f'key_{file_num}.csv'
    path_fsr = os.path.join(source_path, folder_name, fsr_file_name)
    path_key = os.path.join(source_path, folder_name, ppg_file_name)

    return path_fsr, path_key


def get_path_v2(source_path, folder_name, file_num):
    fsr_file_name = f'fsr_{file_num}.csv'
    path_fsr = os.path.join(source_path, folder_name, fsr_file_name)

    return path_fsr


def get_key_v1(path):
    df_key = pd.read_csv(path)
    df_key = df_key.drop('userid', axis=1)

    return df_key


def get_key_v3(path):
    df_key = pd.read_csv(path)
    df_key['timestamp'] = df_key['timestamp'].astype('int64')
    df_key = df_key.rename(columns={'value': "key"})

    return df_key


def get_fsr_v1(path):

    timestamps = []
    datetimes = []
    sensor_fsr_arr = []
    sensor_ppg_arr = []

    processed_data = []
    _processed_data = {}

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=",")

        start = 0
        end = 0
        prev_timestamp = 0
        for i, line in enumerate(reader):
            if len(line) != 18:
                temp = line[0].split(sep=':')
                key = temp[0].strip()
                value = int(temp[1].strip())
                _processed_data[key] = value

                if key == 'PpgHR':
                    _processed_data['timestamp'] = prev_timestamp
                    _processed_data['start'] = start
                    _processed_data['end'] = end
                    processed_data.append(_processed_data)
                    _processed_data = {}
                    start = end + 1
            else:
                timestamp = int(line[0])
                prev_timestamp = timestamp
                lt = time.localtime(timestamp // 1000)
                tf = '%04d%02d%02d-%02d%02d%02d' % (lt.tm_year, lt.tm_mon,
                                                    lt.tm_mday, lt.tm_hour,
                                                    lt.tm_min, lt.tm_sec)
                sensor_fsr = np.array(line[1:17], dtype=int)
                sensor_ppg = int(line[17])

                datetimes.append(tf)
                timestamps.append(timestamp)
                sensor_fsr_arr.append(sensor_fsr)
                sensor_ppg_arr.append(sensor_ppg)

            end += 1

        # print(processed_data)
        df_laxtha = pd.DataFrame(processed_data)

        sensor_fsr_arr = np.array(sensor_fsr_arr)
        sensor_fsr_sum = np.sum(sensor_fsr_arr, axis=1, dtype=int)
        df_sensor = pd.DataFrame({
            'timestamp': timestamps,
            'datetime': datetimes,
            'fsr': list(sensor_fsr_arr),
            'fsr_sum': sensor_fsr_sum,
            'ppg': sensor_ppg_arr
        })

        col = df_laxtha.pop("timestamp")
        df_laxtha.insert(0, col.name, col)

        return df_sensor, df_laxtha


def get_fsr_v2(path):
    df = pd.read_csv(path)
    df = df[['timestamp', 'amp', 'sens', 'fsr']]

    df['fsr'] = df['fsr'].apply(
        lambda x: np.array(x.split(' ')[:32], dtype=int))

    df['fsr_sum'] = np.sum(np.stack(df['fsr'], axis=0), axis=1)

    return df


def get_fsr_v3(path):

    cushion_ts = []
    cushion_fsr = []
    back_ts = []
    back_fsr = []
    ppg_ts = []
    ppg_ir = []
    ppg_red = []

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=",")

        start = 0
        end = 0
        prev_timestamp = 0
        for i, line in enumerate(reader):
            data_name = line[0]
            if data_name == 'CarSeat Cushion':
                ts = int(line[1])
                fsr = line[2].split(sep=' ')
                fsr = [int(x) for x in fsr]

                cushion_ts.append(ts)
                cushion_fsr.append(fsr)

            elif data_name == 'algoD8C0':
                ts = int(line[1])
                ir = int(line[2])
                red = int(line[3])

                ppg_ts.append(ts)
                ppg_ir.append(ir)
                ppg_red.append(red)

            # print(f'{i}: {line}')

    fsr = cushion_fsr
    fsr_ts = cushion_ts
    fsr_sum = np.sum(np.array(fsr), axis=1, dtype=int)
    df_sensor = pd.DataFrame({
        'timestamp': fsr_ts,
        'fsr': fsr,
        'fsr_sum': fsr_sum
    })

    df_ppg = pd.DataFrame({
        'timestamp': ppg_ts,
        'ppg_ir': ppg_ir,
        'ppg_red': ppg_red
    })

    return df_sensor, df_ppg


def get_fsr_v4(path):

    cushion_ts = []
    cushion_fsr = []
    cushion_set = []
    back_ts = []
    back_fsr = []
    back_set = []
    ppg_ts = []
    ppg_ir = []
    ppg_red = []
    key_ts = []
    key_value = []
    loc_ts = []
    loc_lati = []
    loc_long = []
    loc_alti = []
    loc_acc = []
    acc_ts = []
    acc_x = []
    acc_y = []
    acc_z = []

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=",")

        start = 0
        end = 0
        prev_timestamp = 0
        for i, line in enumerate(reader):
            data_name = line[0]
            ts = int(line[1])
            length = int(math.log10(ts)) + 1
            # print("length: ", length)
            if length != 13:
                continue
            # lt = time.localtime(ts // 1000)
            # dt = '%04d%02d%02d-%02d%02d%02d' % (lt.tm_year, lt.tm_mon,
            #                                     lt.tm_mday, lt.tm_hour,
            #                                     lt.tm_min, lt.tm_sec)
            if data_name == 'CarSeat Cushion':
                setting = [int(line[2]), int(line[3])]
                fsr = line[4].split(sep=' ')
                fsr = [int(x) for x in fsr]
                cushion_ts.append(ts)
                cushion_fsr.append(fsr)
                cushion_set.append(setting)

            elif data_name == 'CarSeat Back':
                setting = [int(line[2]), int(line[3])]
                fsr = line[4].split(sep=' ')
                fsr = [int(x) for x in fsr]

                back_ts.append(ts)
                back_fsr.append(fsr)
                back_set.append(setting)

            elif data_name == 'algoD8C0':
                ir = int(line[2])
                red = int(line[3])

                ppg_ts.append(ts)
                ppg_ir.append(ir)
                ppg_red.append(red)

            elif data_name == 'Breathing':
                value = int(line[2])

                key_ts.append(ts)
                key_value.append(value)

            elif data_name == 'Location':
                continue
                loc_ts.append(ts)
                loc_lati.append(float(line[2]))
                loc_long.append(float(line[3]))
                loc_alti.append(float(line[4]))
                loc_acc.append(float(line[5]))

            elif data_name == 'Acceleration':
                continue
                x = float(line[2])
                y = float(line[3])
                z = float(line[4])

                acc_ts.append(ts)
                acc_x.append(x)
                acc_y.append(y)
                acc_z.append(z)

    if cushion_ts:
        fsr_sum = np.sum(np.array(cushion_fsr), axis=1, dtype=int)
    else:
        fsr_sum = 0
    df_sc = pd.DataFrame({
        'timestamp': cushion_ts,
        'fsr': cushion_fsr,
        'fsr_sum': fsr_sum
    })

    if back_ts:
        fsr_sum = np.sum(np.array(back_fsr), axis=1, dtype=int)
    else:
        fsr_sum = 0
    df_sb = pd.DataFrame({
        'timestamp': back_ts,
        'fsr': back_fsr,
        'fsr_sum': fsr_sum
    })

    df_p = pd.DataFrame({
        'timestamp': ppg_ts,
        'ppg_ir': ppg_ir,
        'ppg_red': ppg_red
    })

    df_k = pd.DataFrame({
        'timestamp': key_ts,
        'key': key_value
    })

    df_l = pd.DataFrame({
        'timestamp': loc_ts,
        'latitude': loc_lati,
        'longitude': loc_long,
        'altitude': loc_alti,
        'accuracy': loc_acc
    })

    df_a = pd.DataFrame({
        'timestamp': acc_ts,
        'x': acc_x,
        'y': acc_y,
        'z': acc_z
    })

    data = {
        'cushion': df_sc,
        'back': df_sb,
        'key': df_k,
        'ppg': df_p,
        'location': df_l,
        'acceleration': df_a
    }

    # display(df_sb.head())

    for key, df in data.items():
        df['datetime'] = df['timestamp'].apply(to_datetime)
        reorder_columns(df, ['timestamp', 'datetime'])

    return data


def get_data_v1(source_path, folder_name, file_num):

    path_fsr, path_key = get_path_v1(source_path, folder_name, file_num)
    df_s, df_l = get_fsr_v1(path_fsr)
    df_k = get_key_v1(path_key)

    return df_s, df_k, df_l


def get_data_v2(source_path, folder_name, file_num):
    path_fsr = get_path_v2(source_path, folder_name, file_num)
    # print(path_fsr)
    df_s = get_fsr_v2(path_fsr)

    return df_s


def get_data_v3(source_path, folder_name, file_num):
    path_fsr, path_key = get_path_v1(source_path, folder_name, file_num)
    # print(path_fsr, path_key)
    df_s, df_p = get_fsr_v3(path_fsr)
    df_k = get_key_v3(path_key)

    return df_s, df_k, df_p


def get_data_v4(source_path, folder_name, file_num):
    path_fsr, _ = get_path_v1(source_path, folder_name, file_num)
    # print(path_fsr, path_key)
    return get_fsr_v4(path_fsr)


def get_data(
        version,
        source_path,
        folder_name,
        file_num,
        print_option=False,
):

    if version == 'v1':
        return get_data_v1(source_path, folder_name, file_num)
    elif version == 'v2':
        return get_data_v2(source_path, folder_name, file_num)
    elif version == 'v3':
        return get_data_v3(source_path, folder_name, file_num)
    elif version == 'v4':
        return get_data_v4(source_path, folder_name, file_num)
    else:
        print('Error!! - please insert correct version')

#############################################################################
# Frequency
#############################################################################


def get_fs(df):
    timestamp = df['timestamp'].to_numpy()
    fs = int((len(timestamp) / (timestamp[-1] - timestamp[0]))*1000)
    # print(f'freq: {fs}')
    return fs

#############################################################################
# Slice
#############################################################################


def slice_df_based_on_time_v1(
    df,
    slice_index=[None, None],
    fs=128,
):
    """
    slice df based on "time(s)"
    slice from df.iloc[0] to df.iloc[start]
    and from df.iloc[-1] to df.iloc[-end]
    """
    start = int(slice_index[0]*fs)
    end = int(slice_index[1]*fs)
    if start is not None and end is not None:
        df = df.iloc[start:-end, :]
    elif start is not None and end is None:
        df = df.iloc[start:, :]
    elif start is None and end is not None:
        df = df.iloc[:-end, :]

    return df


def slice_df_based_on_time_v2(
    df,
    slice_index=[None, None],
    fs=128,
):
    """
    slice df based on absolute "time(s)"
    """
    start = slice_index[0]
    end = slice_index[1]
    start = int(slice_index[0]*fs)
    end = int(slice_index[1]*fs)
    if start is not None and end is not None:
        df = df.iloc[start:end, :]
    elif start is not None and end is None:
        df = df.iloc[start:, :]
    elif start is None and end is not None:
        df = df.iloc[:end, :]

    return df


def slice_df_based_on_idx(
    df,
    slice_index=[None, None],
    fs=128,
):
    df = copy.deepcopy(df)
    """
    slice df based on idx
    """
    start = slice_index[0]
    end = slice_index[1]

    if start is not None and end is not None:
        df = df.iloc[start:end, :]
    elif start is not None and end is None:
        df = df.iloc[start:, :]
    elif start is None and end is not None:
        df = df.iloc[:end, :]

    return df


#############################################################################
# Match
#############################################################################


def match_ts_OnKey(
    df_s,
    df_k
):
    """
    Args:
        df_s (dataframe): df sensor
        df_k (dataframe): df key
    """
    print(df_s.timestamp.min(), df_s.timestamp.max())
    print(df_k.timestamp.min(), df_k.timestamp.max())

    __min = df_k.timestamp.min()
    __max = df_k.timestamp.max()
    df_s = df_s[(df_s.timestamp >= __min) & (df_s.timestamp <= __max)]
    # print(f'matched df_s: {df_s.shape[0]}')

    df_s.index = df_s.timestamp
    df_k = df_k.set_index('timestamp')
    print(df_s.index)
    df_key_re = df_k.reindex(df_s.index, method='ffill')  # nearest

    df_s['key'] = df_key_re['key']
    df_s = df_s.dropna(subset=['key']).reset_index(drop=True)

    return df_s


def match_ts(
    df1,
    df2,
    base=None
):
    if base not in [1, 2, None]:
        return print('Wrong base parameter (only 1 or 2)')

    __min1 = df1.timestamp.min()
    __max1 = df1.timestamp.max()
    __time1 = __max1 - __min1

    __min2 = df2.timestamp.min()
    __max2 = df2.timestamp.max()
    __time2 = __max2 - __min2

    if __time1 < __time2 or base == 1:
        # print('df1 < df2: matched timestamp based on df1')
        __min = __min1
        __max = __max1
    elif __time1 > __time2 or base == 2:
        # print('df1 > df2: matched timestamp based on df2')
        __min = __min2
        __max = __max2

    df1 = df1[(df1.timestamp >= __min) & (df1.timestamp <= __max)]
    df2 = df2[(df2.timestamp >= __min) & (df2.timestamp <= __max)]

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    return df1, df2


#############################################################################
# Interpolation
#############################################################################


def interpolate_df(df, col_name, mean_limit_ratio,
                   interval):

    _df = df[col_name]
    length = len(_df)

    increment = interval//2
    num = length//increment

    interp_sig = np.zeros((length))
    start, end = [0, interval]
    for i in range(0, num+1):
        # start = interval*i
        # end = interval*(i+1)-1

        # if start > margin:
        #     start -= margin
        #     end += margin

        # if end > length:
        #     start = length - interval
        #     end = length

        sig = _df.iloc[start:end]
        mean = int(sig.mean())

        margin = int(mean*mean_limit_ratio)
        error = abs(mean-sig)

        count = sig.loc[error > margin]

        # print(count.shape)

        text = f'range No {i} :{start} ~ {end} {length} {interval} -> {count.shape}'
        # print(text)

        sig.loc[error > margin] = np.nan
        sig = sig.interpolate(method='linear', axis=0)

        interp_sig[start: end] = sig

        start += increment
        end += increment

    interp_sig[end:] = _df.iloc[end:]
    df[col_name] = interp_sig

    return interp_sig


def interpolate_each_sensor(df, mean_limit_ratio=2, interval=2048):
    fsr = np.stack(df['fsr'], axis=0)
    N, num_of_sensors = fsr.shape
    # print(f'fsr shape: {fsr.shape}')
    # print(num_of_sensors)

    interp_fsr = []
    for i in range(num_of_sensors):
        # print(f'start sensor {i} ================')
        sig = fsr[:, i]
        df['fsr_temp'] = sig

        # if True:
        #     plot_simple(sig, f'{i} before')
        interp_sig = interpolate_df(
            df, 'fsr_temp',  mean_limit_ratio, interval)
        interp_fsr.append(interp_sig)

        # print(interp_sig.shape)
        # if True:
        #     plot_simple(interp_sig, f'{i} after')
        # display(df.head(3))

    interp_fsr = np.stack(interp_fsr, axis=1)
    new_fsr_sum = np.sum(interp_fsr, axis=1)
    df['fsr'] = [row for row in interp_fsr]
    df['fsr_sum'] = new_fsr_sum
    df = df.drop(['fsr_temp'], axis=1)
    return df

#############################################################################
# sensor selection
#############################################################################


def select_sensors(df, mean_limit_ratio=0.5, std_limit_ratio=2):
    fsr = np.stack(df['fsr'].values, axis=0)
    _, num_of_sensors = fsr.shape

    mean_all = int(np.mean(fsr))
    std_all = int(np.std(fsr))

    std_limit = int(std_all*std_limit_ratio)
    mean_limit = int(mean_all*mean_limit_ratio)

    selected_idxs = []
    for i in range(num_of_sensors):
        sig = fsr[:, i]

        std = int(np.std(sig))
        mean = int(np.mean(sig))

        check = False
        if mean > mean_limit and std < std_limit:
            selected_idxs.append(i)
            check = True

        # text = f'No.{i}~({mean},{std})vs({mean_limit},{std_limit}) {check}'
        # plot_simple(sig, title=text)

    new_fsr = fsr[:, selected_idxs]
    new_fsr_sum = np.sum(new_fsr, axis=1)
    df['fsr'] = [row for row in new_fsr]
    df['fsr_sum'] = new_fsr_sum

    # print(f'selected_idxs: {selected_idxs}')

    return df, selected_idxs


#############################################################################
# data moving window
#############################################################################


def get_indice(df, interval):
    N, _ = df.shape
    increment = interval//4
    num = N//increment

    start = []
    end = []

    start_ind = 0
    end_ind = interval
    for i in range(1, num+1):
        # print(f'{i}: {start_ind} ~ {end_ind}')
        start.append(start_ind)
        end.append(end_ind)

        start_ind += increment
        end_ind += increment
        if end_ind > N:
            break

    df_indices = pd.DataFrame({
        'start': start,
        'end': end,
    })

    return df_indices


#############################################################################
# merge df
#############################################################################


def merge_fsr_cols(df):
    fsrx = np.stack(df['fsr_x'].values, axis=0)
    fsry = np.stack(df['fsr_y'].values, axis=0)
    fsr = np.concatenate((fsrx, fsry), axis=1)
    fsr_sum = fsr.sum(axis=1)

    df['fsr'] = [x for x in fsr]
    df['fsr_sum'] = fsr_sum
    df = df.drop(['fsr_x', 'fsr_y'], axis=1)
    df['datetime'] = df['timestamp'].apply(to_datetime)
    df = df.drop(['datetime_x', 'datetime_y'], axis=1)
    return df


def merge_fsr_dfs(df1, df2):

    for i, df in enumerate([df1, df2]):
        num = i+1
        df = df.drop_duplicates(subset='datetime')
        df = df.set_index('datetime')
        df = df.resample('10L', label='left',
                         closed='left', base=0).ffill()
        df = df.reset_index()

        df['timestamp'] = df['datetime'].apply(
            lambda x: int(datetime.datetime.timestamp(x)*1000))
        # df = df.drop(['datetime', 'fsr_sum'], axis=1)
        df = df.drop(['fsr_sum'], axis=1)

        if num == 1:
            df1 = df
        elif num == 2:
            df2 = df

    df = df1.merge(df2, how='inner', on='timestamp')
    df = df.dropna().reset_index(drop=True)
    df = merge_fsr_cols(df)
    return df


def resample_df(df, sampling_rate='10L'):
    df = df.drop_duplicates(subset='datetime')
    df = df.set_index('datetime')
    df = df.resample(sampling_rate, label='left',
                     closed='left', base=0).ffill()
    df = df.reset_index()

    df['timestamp'] = df['datetime'].apply(
        lambda x: int(datetime.datetime.timestamp(x)*1000))

    return df


def merge_extra_dfs(df_s, *dfs):
    df_s = resample_df(df_s)
    for df in dfs:
        df = resample_df(df)
        df_s = df_s.merge(df, how='inner', on='timestamp')
        display(df_s.head(3))

    df_s = df_s.dropna().reset_index(drop=True)

    return df_s
