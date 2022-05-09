# -*- coding: utf-8 -*-

import pandas as pd
from dramkit.fintools.fintools import Boll


def boll_label_uplow(data, tol=0.2/100, judge_type='maxmin'):
    '''
    | 根据布林带，构造潜在高低点标签
    | 思路：
    |     最低价低于布林带下轨的被标注为潜在低点（less_low）
    |     最高价高于布林带上轨的被标注为潜在高点（more_up）
    |     上一个低点之后的所有潜在高点中的最高值或最后一个值标为实际高点
    |     上一个高点之后的所有潜在低点中的最低值或最后一个值标为实际低点
    |     （判断方式由judge_type指定，judge_type可取'maxmin'或'last'）
    | 
    | data须包含['high', 'low', 'close', 'boll_low', 'boll_up']列
    | tol指定潜在高低点判断时距离布林带上下轨的距离容忍度
    |
    | 返回结果包含列：
    | ['high', 'low', 'close', 'boll_low', 'boll_up', 'less_low', 'more_up', 'label']，
    | 其中
    | less_low标注最低价是否低于布林带下轨，
    | more_up标注最高价是否高于布林带上轨，
    | label标注是否为阶段高低点
    '''

    data['less_low'] = data[['low', 'boll_low']].apply(lambda x:
                    -1 if x['low'] <= x['boll_low']*(1+tol) else 0, axis=1)
    data['more_up'] = data[['high', 'boll_up']].apply(lambda x:
                    1 if x['high'] >= x['boll_up']*(1-tol) else 0, axis=1)

    data['label_all'] = data['less_low'] + data['more_up']

    if judge_type == 'last':
        tmp = data[data['label_all'] != 0][['label_all']]
        tmp['label'] = tmp['label_all'].diff().shift(-1)
        tmp['label'] = tmp['label'].fillna(0).replace({2: -1, -2: 1})
        tmp.loc[tmp.index[-1], 'label'] = tmp.loc[tmp.index[-1], 'label_all']


    elif judge_type == 'maxmin':
        # max_col, min_col = 'high', 'low'
        max_col, min_col = 'close', 'close'
        maxmin_cols = list(set([max_col, min_col]))

        tmp = data[data['label_all'] != 0][maxmin_cols + ['label_all']]
        tmp['label'] = tmp['label_all']
        k = 0
        while k < tmp.shape[0]:
            k1 = k

            if tmp.loc[tmp.index[k], 'label_all'] == 1:
                while k1 < tmp.shape[0] and \
                                    tmp.loc[tmp.index[k1], 'label_all'] != -1:
                    k1 += 1
                sure = False
                for n in range(k, k1):
                    if tmp.loc[tmp.index[n], max_col] == \
                         tmp.loc[tmp.index[k:k1], max_col].max() and not sure:
                        tmp.loc[tmp.index[n], 'label'] = 1
                        sure = True
                    else:
                        tmp.loc[tmp.index[n], 'label'] = 0

            if tmp.loc[tmp.index[k], 'label_all'] == -1:
                while k1 < tmp.shape[0] and \
                                    tmp.loc[tmp.index[k1], 'label_all'] != 1:
                    k1 += 1
                sure = False
                for n in range(k, k1):
                    if tmp.loc[tmp.index[n], min_col] == \
                         tmp.loc[tmp.index[k:k1], min_col].min() and not sure:
                        tmp.loc[tmp.index[n], 'label'] = -1
                        sure = True
                    else:
                        tmp.loc[tmp.index[n], 'label'] = 0

            k = k1

    data['label'] = tmp['label']
    data['label'] = data['label'].fillna(0)

    return data


def boll_label_uplow_cross(data, tol=0.0/100, judge_type='maxmin'):
    '''
    | 根据布林带，构造潜在高低点标签
    | 思路：
    |     从下往上穿过布林带下轨被标注为潜在买点（low_buy）
    |     从上往下穿过布林带下轨被标注为潜在卖点（up_sel）
    |     上一个买点之后的所有潜在卖点中的最高值或最后一个值标为实际卖点
    |     上一个卖点之后的所有潜在买点中的最低值或最后一个值标为实际买点
    |     （判断方式由judge_type指定，judge_type可取'maxmin'或'last'）
    |
    | data须包含['high', 'low', 'close', 'boll_low', 'boll_up']列
    | tol指定潜在高低点判断时距离布林带上下轨的距离容忍度
    |
    | 返回结果包含列：
    | ['high', 'low', 'close', 'boll_low', 'boll_up', 'low_buy', 'up_sel', 'label']，
    | 其中
    | low_buy标注潜在买点，
    | more_up标注潜在卖点，
    | label标注是否为阶段买卖点
    '''

    # 潜在买点
    data['less_low'] = data[['low', 'boll_low']].apply(lambda x:
                        -1 if x['low'] < x['boll_low']*(1+tol) else 0, axis=1)
    # data['less_low'] = data[['close', 'boll_low']].apply(lambda x:
    #                 -1 if x['close'] < x['boll_low']*(1+tol) else 0, axis=1)
    data['less_low_'] = data['less_low'].shift(1)
    data.loc[data.index[0], 'less_low_'] = 0
    data['more_low'] = data[['low', 'boll_low']].apply(lambda x:
                        -1 if x['low'] >= x['boll_low']*(1-tol) else 0, axis=1)
    # data['more_low'] = data[['close', 'boll_low']].apply(lambda x:
    #                 -1 if x['close'] >= x['boll_low']*(1-tol) else 0, axis=1)
    data['low_buy'] = data['more_low'] + data['less_low_']
    data['low_buy'] = data['low_buy'].replace({-2: -1, -1: 0})
    data.drop(['less_low', 'less_low_', 'more_low'], axis=1, inplace=True)

    # 潜在卖点
    data['more_up'] = data[['high', 'boll_up']].apply(lambda x:
                        1 if x['high'] > x['boll_up']*(1-tol) else 0, axis=1)
    # data['more_up'] = data[['close', 'boll_up']].apply(lambda x:
    #                   1 if x['close'] > x['boll_up']*(1-tol) else 0, axis=1)
    data['more_up_'] = data['more_up'].shift(1)
    data.loc[data.index[0], 'more_up_'] = 0
    data['less_up'] = data[['high', 'boll_up']].apply(lambda x:
                        1 if x['high'] <= x['boll_up']*(1+tol) else 0, axis=1)
    # data['less_up'] = data[['close', 'boll_up']].apply(lambda x:
    #                   1 if x['close'] <= x['boll_up']*(1+tol) else 0, axis=1)
    data['up_sel'] = data['less_up'] + data['more_up_']
    data['up_sel'] = data['up_sel'].replace({2: 1, 1: 0})
    data.drop(['more_up', 'more_up_', 'less_up'], axis=1, inplace=True)

    data['label_all'] = data['low_buy'] + data['up_sel']

    if judge_type == 'last':
        tmp = data[data['label_all'] != 0][['label_all']]
        tmp['label'] = tmp['label_all'].diff().shift(-1)
        tmp['label'] = tmp['label'].fillna(0).replace({2: -1, -2: 1})
        tmp.loc[tmp.index[-1], 'label'] = tmp.loc[tmp.index[-1], 'label_all']

    elif judge_type == 'maxmin':
        # max_col, min_col = 'high', 'low'
        max_col, min_col = 'close', 'close'
        maxmin_cols = list(set([max_col, min_col]))

        tmp = data[data['label_all'] != 0][maxmin_cols + ['label_all']]
        tmp['label'] = tmp['label_all']
        k = 0
        while k < tmp.shape[0]:
            k1 = k

            if tmp.loc[tmp.index[k], 'label_all'] == 1:
                while k1 < tmp.shape[0] and \
                                    tmp.loc[tmp.index[k1], 'label_all'] != -1:
                    k1 += 1
                sure = False
                for n in range(k, k1):
                    if tmp.loc[tmp.index[n], max_col] == \
                         tmp.loc[tmp.index[k:k1], max_col].max() and not sure:
                        tmp.loc[tmp.index[n], 'label'] = 1
                        sure = True
                    else:
                        tmp.loc[tmp.index[n], 'label'] = 0

            if tmp.loc[tmp.index[k], 'label_all'] == -1:
                while k1 < tmp.shape[0] and \
                                    tmp.loc[tmp.index[k1], 'label_all'] != 1:
                    k1 += 1
                sure = False
                for n in range(k, k1):
                    if tmp.loc[tmp.index[n], min_col] == \
                         tmp.loc[tmp.index[k:k1], min_col].min() and not sure:
                        tmp.loc[tmp.index[n], 'label'] = -1
                        sure = True
                    else:
                        tmp.loc[tmp.index[n], 'label'] = 0

            k = k1

    data['label'] = tmp['label']
    data['label'] = data['label'].fillna(0)

    return data


def get_boll_label_uplow(data, lag=15, width=2, tol=0.2/100,
                         judge_type='maxmin', n_dot=3):
    '''
    | 生成基于布林带的高低点标签
    | data中必须包含['close', 'high', 'low']三列
    | lag、width、n_dot参数同 :func:`dramkit.fintools.fintools.Boll` 函数
    | judge_type、tol参数同 :func:`boll_label_uplow` 函数
    '''

    # 布林带
    boll_close = Boll(data['close'], lag=lag, width=width, n_dot=n_dot)

    # 布林带高低点标签
    # df_boll = boll_close.drop(['boll_mid', 'boll_std'], axis=1)
    df_boll = boll_close.copy()
    df_label = pd.merge(data[['low', 'high']], df_boll, how='left',
                        left_index=True, right_index=True)

    df_label = boll_label_uplow(df_label, tol=tol, judge_type=judge_type)

    return df_label['label'], df_label['label_all']


def get_boll_label_uplow_cross(data, lag=15, width=2, tol=0.0/100,
                              judge_type='maxmin', n_dot=3):
    '''
    | 生成基于布林带穿越的高低点标签
    | data中必须包含['close', 'high', 'low']三列
    | lag、width、n_dot参数同 :func:`dramkit.fintools.fintools.Boll` 函数
    | judge_type、tol参数同 :func:`boll_label_uplow_cross` 函数
    '''

    # 布林带
    boll_close = Boll(data['close'], lag=lag, width=width, n_dot=n_dot)

    # 布林带高低点标签
    # df_boll = boll_close.drop(['boll_mid', 'boll_std'], axis=1)
    df_boll = boll_close.copy()
    df_label = pd.merge(data[['low', 'high']], df_boll, how='left',
                        left_index=True, right_index=True)

    df_label = boll_label_uplow_cross(df_label, tol=tol, judge_type=judge_type)

    return df_label['label'], df_label['label_all']


if __name__ == '__main__':
    from dramkit import load_csv, plot_series
    from dramkit.plottools.plot_candle import plot_candle
    from dramkit.fintools.finplot import plot_boll
    from dramkit.fintools.utils_gains import get_yield_curve
    import time
    strt_tm = time.time()

    fpath = '../test/510050_daily_pre_fq.csv'

    data = load_csv(fpath).rename(columns={'date': 'time'})
    data = data.reindex(columns=['time', 'open', 'low', 'high', 'close',
                                 'volume', 'amount'])

    # 布林带
    boll_close = Boll(data['close'], lag=15, width=2, n_dot=3)
    df_boll = pd.merge(boll_close.drop('close', axis=1), data, how='right',
                       left_index=True, right_index=True).reset_index()
    df_boll = df_boll.reindex(columns=['time', 'open', 'high', 'low', 'close',
                                       'boll_up', 'boll_low', 'boll_mid'])
    plot_boll(df_boll, N=100, figsize=(11, 7))


    # 布林带高低点
    lag, width, tol = 15, 2, 0.0/100
    data['label'], data['label_all'] = get_boll_label_uplow(data, lag=lag,
                          width=width, tol=tol, judge_type='maxmin', n_dot=3)
    plot_series(data.iloc[-300:,], {'close': '.-k'},
                cols_to_label_info={'close':
                          [['label', (-1, 1), ('r^', 'bv'), ('Max', 'Min')],
                           ['label_all', (-1, 1), ('m.', 'g.'), ('low', 'up')]
                          ]},
                  figsize=(11, 7), title='50ETF择时信号标签（高低点）')

    # 布林带上下轨穿越买卖点
    data['label'], data['label_all'] = get_boll_label_uplow_cross(data,
                  lag=lag, width=width, tol=tol, judge_type='maxmin', n_dot=3)
    plot_series(data.iloc[-300:,], {'close': '.-k'},
                cols_to_label_info={'close':
                          [['label', (-1, 1), ('r^', 'bv'), ('Max', 'Min')],
                           ['label_all', (-1, 1), ('m.', 'g.'), ('low', 'up')]
                          ]},
                  figsize=(11, 7), title='50ETF择时信号标签（高低点穿越）')
    data['label'] = data['label'].shift(1)
    data['label_all'] = data['label_all'].shift(1)
    data['label'] = data['label'].fillna(0)
    data['label_all'] = data['label_all'].fillna(0)
    plot_candle(data.iloc[-500:, :],
                cols_to_label_info={
                    'close': [['label_all', (-1, 1), ('m.', 'b.'), False],
                              ['label', (-1, 1), ('m^', 'bv'), ('买', '卖')]],},
                args_ma=None, args_boll=[lag, width], alpha=0.4, width=0.5,
                plot_below=None, args_ma_below=None, figsize=(11, 7),
                markersize=12)

    # data = data[data['time'] >= '2016-01-01'].copy()
    # data.set_index('time', inplace=True)
    # trade_gain_info, df_gain = get_yield_curve(data, 'label',
    #                                         col_price='close',
    #                                         col_price_buy='close',
    #                                         col_price_sel='close',
    #                                         baseMny=1000, baseVol=None,
    #                                         VolF_add='base_1',
    #                                         VolF_sub='hold_base_1',
    #                                         fee=1.5/1000, max_loss=None,
    #                                         max_gain=None, max_down=None,
    #                                         VolF_stopLoss=0, init_cash=0,
    #                                         forceFinal0='settle')

    # trade_gain_info, df_gain = get_yield_curve(data, 'label_all',
    #                                         col_price='close',
    #                                         col_price_buy='close',
    #                                         col_price_sel='close',
    #                                         baseMny=1000, baseVol=None,
    #                                         VolF_add='base_1',
    #                                         VolF_sub='hold_base_1',
    #                                         fee=1.5/1000, max_loss=None,
    #                                         max_gain=None, max_down=None,
    #                                         VolF_stopLoss=0, init_cash=0,
    #                                         forceFinal0='settle')


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
