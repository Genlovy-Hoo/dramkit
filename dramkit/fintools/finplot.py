# -*- coding: utf-8 -*-

import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def plot_boll(df_boll, N=100, figsize=(12.5, 9), markersize=10):
    '''
    | 绘制布林带
    | df_boll须包含以下列:
    |     ['time', 'open', 'high', 'low', 'close', 'boll_up', 'boll_low', 'boll_mid']
    '''
    df = df_boll.iloc[-N:, :].copy()
    df['tmp'] = df.apply(
                    lambda x: 1 if x['close'] >= x['open'] else 0, axis=1)
    df.reset_index(drop=True, inplace=True)

    plt.figure(figsize=figsize)

    # 布林带三线
    plt.plot(df['boll_low'], '-r')
    plt.plot(df['boll_up'], '-g')
    plt.plot(df['boll_mid'], '-y')

    # 最高最低价标注
    plt.plot(df['high'], '.k')
    plt.plot(df['low'], '.k')

    # 收盘高于开盘画红竖线
    close_big_open = df.query('tmp == 1')
    plt.plot(close_big_open['close'], '_r', markersize=markersize)
    plt.plot(close_big_open['open'], '_r', markersize=markersize)
    for idx in close_big_open.index:
        ymin = close_big_open.loc[idx, 'low']
        ymax = close_big_open.loc[idx, 'high']
        plt.vlines(idx, ymin, ymax, color='r', linewidth=2)

    # 收盘低于开盘画绿竖线
    close_sml_open = df.query('tmp == 0')
    plt.plot(close_sml_open['close'], '_g', markersize=markersize)
    plt.plot(close_sml_open['open'], '_g', markersize=markersize)
    for idx in close_sml_open.index:
        ymin = close_sml_open.loc[idx, 'low']
        ymax = close_sml_open.loc[idx, 'high']
        plt.vlines(idx, ymin, ymax, color='g', linewidth=2)

    xpos = [int(x*N/8) for x in range(0, 8)] + [N-1]
    plt.xticks(xpos, [df['time'].iloc[x] for x in xpos])

    plt.show()


if __name__ == '__main__':
    from dramkit import load_csv
    from dramkit.fintools.fintools import Boll
    
    fpath = '../test/510050_daily_pre_fq.csv'

    data = load_csv(fpath).rename(columns={'date': 'time'}).set_index('time')
    data = data.reindex(columns=['code', 'last_close', 'open', 'low', 'high',
                                 'close', 'change', 'change_pct', 'volume',
                                 'amount'])

    # 布林带
    boll_close = Boll(data['close'], lag=15, width=2, n_dot=3)
    df_boll = pd.merge(boll_close.drop('close', axis=1), data, how='right',
                       left_index=True, right_index=True).reset_index()
    df_boll = df_boll.reindex(columns=['time', 'open', 'high', 'low', 'close',
                                       'boll_up', 'boll_low', 'boll_mid'])
    plot_boll(df_boll, figsize=(10.5, 7))
