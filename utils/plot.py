import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

FONTSIZE = 20


def print_dict(dic):
    for k, v in dic.items():
        print(f'{k:15s}: {v:10.2f}')


def display_np(np):

    display(df.head(3))


def display_df(df):
    display(df.head(3))


def plot_simple(
    y,
    title='None'
):
    plt.figure(figsize=(16, 3))
    plt.plot(y)
    plt.title(title, fontsize=FONTSIZE)
    plt.grid(linestyle='--')
    plt.xticks(range(0, len(y)+500, 500))
    plt.show()


# def plot_each_sensor(
#     df,
#     num_of_sensors=16,
#     title='0000',
#     limit_mean=10000,
#     limit_std=2000,
# ):
#     sig = np.stack(df.values, axis=0)
#     print(f'sig shape: {sig.shape}')

#     selected_sensor = []
#     for i in range(num_of_sensors):
#         sensor_sig = sig[:, i]
#         # print(sensor_sig.shape)
#         std = np.round(np.std(sensor_sig), 2)
#         mean = np.round(np.mean(sensor_sig), 2)

#         check = False
#         if std < limit_std and mean > limit_mean:
#             check = True
#             selected_sensor.append(i)

#         text = f'{title} FSR No.{i} mean:{mean} std:{std} check:{check}'
#         plot_simple(
#             sensor_sig,
#             title=text,
#         )

def plot_each_sensor(
    df,
    title="XXXX"
):
    sig_all = np.stack(df['fsr'].values, axis=0)
    _, num_of_sensors = sig_all.shape

    mean_all = int(np.mean(sig_all))
    std_all = int(np.std(sig_all))

    std_limit = int(std_all*2)
    mean_limit = int(mean_all*0.5)

    for i in range(num_of_sensors//5):
        sig = sig_all[:, i]

        std = int(np.std(sig))
        mean = int(np.mean(sig))

        if std < std_limit and mean > mean_limit:
            check = True
        else:
            check = False

        text = f'{title} No.{i}~({mean},{std})vs({mean_all},{std_all}) {check}'
        plot_simple(sig, title=text)


def plot_one_column(
    df,
    col_name,
    fs=128,
    file_num='0000',
    tag='None',
):
    data = df[col_name]
    N, _ = df.shape
    # total_time = (df['timestamp'][N-1] - df['timestamp'][0])/1000

    t = np.arange(len(data)) / fs
    plt.figure(figsize=(16, 3))
    plt.plot(t, data)
    title = f'[{file_num}][{tag}] {col_name} '
    plt.title(title, fontsize=FONTSIZE)
    plt.show()


def plot_key(df, fs=128):
    print('plot_key')
    data = df['key']
    N, _ = df.shape
    t = np.arange(len(data)) / fs
    plt.figure(figsize=(16, 3))
    plt.plot(t, data)
    plt.show()
