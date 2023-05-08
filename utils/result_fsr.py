import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

from functools import partial

from lib.utils import reorder_columns


plt.rcParams['figure.figsize'] = (16, 3)


COMPARISON_COLS = [
    [['resp_ref_key', 'resp_peak_fsr'], [
        'error_resp_peak', 'mae_resp_peak', 'mape_resp_peak']],
    [['resp_ref_key', 'resp_freq_fsr'], [
        'error_resp_freq', 'mae_resp_freq', 'mape_resp_freq']]]

COMPARISON_COLS_HR = [
    [['resp_ref_key', 'resp_peak_fsr'], [
        'error_resp_peak', 'mae_resp_peak', 'mape_resp_peak']],
    [['resp_ref_key', 'resp_freq_fsr'], [
        'error_resp_freq', 'mae_resp_freq', 'mape_resp_freq']],
    [['hr_freq_ppg', 'hr_freq_fsr'],
        ['error_hr_freq', 'mae_hr_freq', 'mape_hr_freq']]]


COL_ORDER = ['file_num', 'seat_part', 'range', 'start', 'end',
             'mape_resp_peak', 'mape_resp_freq']

COL_ORDER_HR = ['file_num', 'seat_part', 'range', 'start', 'end',
                'mape_resp_peak', 'mape_resp_freq', 'mape_hr_freq']


def get_result_df(df, isHR, isEachSensor):
    def calculate_MAE_df(df, col1, col2):
        y_true, y_pred = df[col1], df[col2]
        error = np.round(y_true - y_pred, 2).astype(int)

        mae = np.round(np.abs(error), 2).astype(int)
        percentage = (np.abs(error)/y_true) * 100
        mape = np.round(percentage, 2)

        return pd.Series([error, mae, mape])

    if isHR:
        comparison_cols = COMPARISON_COLS_HR
    else:
        comparison_cols = COMPARISON_COLS

    # MAE and MAPE
    for in_cols, out_cols in comparison_cols:
        func = partial(calculate_MAE_df, col1=in_cols[0], col2=in_cols[1])
        df[[out_cols[0], out_cols[1], out_cols[2]]
           ] = df.apply(lambda x: func(x), axis=1)
        df[out_cols[0]] = df[out_cols[0]].astype(int)
        df[out_cols[1]] = df[out_cols[1]].astype(int)

    # 100 - MAPE for barplot
    df['p_resp_peak'] = 100 - df['mape_resp_peak']
    df['p_resp_freq'] = 100 - df['mape_resp_freq']
    if isHR:
        df['p_hr_freq'] = 100 - df['mape_hr_freq']

    df['range_detail'] = '('+df['start'].astype(str) + \
        ', ' + df['end'].astype(str) + ')'

    # print(df['range_detail'])
    # display(df)
    if isHR:
        col_order = COL_ORDER_HR
    else:
        col_order = COL_ORDER

    if isEachSensor:
        col_order.insert(2, 'fsr_num')

    reorder_columns(df, col_order)

    return df


def get_result_dic(dic, isHR=True):
    def calculate_MAE_dic(dic, col1, col2):
        y_true, y_pred = dic[col1], dic[col2]
        error = np.round(y_true - y_pred, 2)

        mae = np.round(np.abs(error), 2)
        percentage = (np.abs(error)/y_true) * 100
        mape = np.round(percentage, 2)
        return [error, mae, mape]

    if isHR:
        comparison_cols = COMPARISON_COLS_HR
    else:
        comparison_cols = COMPARISON_COLS

    for in_cols, out_cols in comparison_cols:
        # print(in_cols, out_cols)
        l_res = calculate_MAE_dic(dic, in_cols[0], in_cols[1])
        # print(l_res)
        dic[out_cols[0]] = l_res[0]
        dic[out_cols[1]], dic[out_cols[2]] = l_res[1], l_res[2]

    return dic


def save_csv(df, path, file_name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = f'{timestr}_{file_name}.csv'
    path = os.path.join(path, file_name)
    print(path)
    try:
        df.to_csv(path, index=False)
    except FileNotFoundError:
        print('There is No directory')


def show_result(
    df_s,
    df_r,
    isHR,
    title='None',
):

    RESP_PEAK_LIMIT = 10
    RESP_FREQ_LIMIT = 10
    HR_FREQ_LIMIT = 15

    FONTSIZE = 20

    sig_y = df_s['fsr_sum']

    max_y = sig_y.max()
    min_y = sig_y.min()
    width_arrow = (max_y - min_y)//10

    # line graph
    plt.plot(sig_y)
    plt.title(title, fontsize=FONTSIZE)
    plt.grid(linestyle='--')
    plt.xticks(range(0, len(sig_y)+500, 500))

    currentaxis = plt.gca()

    MARGIN = 50
    # df_resp = df_r[(df_r['mape_resp_peak'] < RESP_PEAK_LIMIT) |
    #    (df_r['mape_resp_freq'] < RESP_FREQ_LIMIT)]
    df_resp = df_r[(df_r['mape_resp_freq'] < RESP_FREQ_LIMIT)]
    for idx, row in df_resp.iterrows():
        if row['range'] == 'all':
            continue

        start = row['start']
        end = row['end']
        x = start
        w = end - start
        h = int(max_y - min_y) + 2*MARGIN
        y = min_y - MARGIN
        currentaxis.add_patch(Rectangle((x, y), w, h,
                                        ec='blue', fc="none",
                                        ls='--', lw=2))
        currentaxis.add_patch(
            Arrow(x, max_y, w, 0, width=width_arrow,
                  ec='lightgray', fc="lightgreen"))

    # if isHR:
    #     df_hr = df_r[(df_r['mape_hr_freq'] < HR_FREQ_LIMIT)]
    #     for idx, row in df_hr.iterrows():
    #         if row['range'] == 'all':
    #             continue
    #         start = row['start']
    #         end = row['end']
    #         x = start
    #         w = end - start
    #         h = int(max_y - min_y) + 2*MARGIN
    #         y = min_y - MARGIN
    #         currentaxis.add_patch(Rectangle((x, y), w, h,
    #                                         ec='green', fc="none",
    #                                         ls='--', lw=2))
    #         currentaxis.add_patch(
    #             Arrow(x, max_y, w, 0, width=width_arrow,
    #                   ec='lightgray', fc="lightgreen"))

    #     df_both = df_r[(df_r['mape_resp_peak'] < RESP_PEAK_LIMIT) | (
    #         df_r['mape_resp_freq'] < RESP_FREQ_LIMIT) & (
    #             df_r['mape_hr_freq'] < HR_FREQ_LIMIT)]

    #     for idx, row in df_both.iterrows():
    #         if row['range'] == 'all':
    #             continue
    #         start = row['start']
    #         end = row['end']
    #         x = start
    #         w = end - start
    #         h = int(max_y - min_y) + 2*MARGIN
    #         y = min_y - MARGIN
    #         currentaxis.add_patch(Rectangle((x, y), w, h,
    #                                         ec='red', fc="none",
    #                                         ls='--', lw=2))

    plt.show()

    values_vars = ['p_resp_peak', 'p_resp_freq']
    if isHR:
        values_vars.append('p_hr_freq')
    _df_r = df_r.melt(id_vars=['range_detail'], value_vars=values_vars,
                      var_name='evaluation', value_name='value')

    if barplot := True:
        df_r = df_r[df_r['range'] == 'partial']
        draw_barplot_condition(df_r, x='range_detail',
                               y='p_resp_peak', title='resp_peak')
        draw_barplot_condition(df_r, x='range_detail',
                               y='p_resp_freq', title='resp_freq')
        if isHR:
            draw_barplot_condition(df_r, x='range_detail',
                                   y='p_hr_freq', title='hr_freq')
        draw_barplot(_df_r, x='range_detail',
                     y='value', hue='evaluation', title='Range analysis')


def get_color_list(df, col_name):
    color_list = []
    for x in df[col_name]:
        if x > 90:
            color_list.append('red')
        elif x > 80:
            color_list.append('coral')
        elif x > 70:
            color_list.append('mistyrose')
        else:
            color_list.append('lightgray')
    return color_list


def draw_barplot_condition(df, x, y, title):
    color_list = get_color_list(df, y)
    plt.xticks(rotation=45)
    ax = sns.barplot(data=df, x=x, y=y, palette=color_list)
    plt.title(title)
    plt.show()


def draw_barplot(df, x, y, title, hue=None):
    # color_list = get_color_list(df, y)
    plt.xticks(rotation=45)
    ax = sns.barplot(data=df, x=x, y=y, hue=hue)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # plt.legend(loc='upper right')
    plt.title(title)
    plt.show()
