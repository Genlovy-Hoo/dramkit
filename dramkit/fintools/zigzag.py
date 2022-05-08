# -*- coding: utf-8 -*-

import numpy as np
from dramkit.plottools.plot_candle import plot_candle


def zigzag(data, high_col='high', low_col='low',
           up_pct=1/100, down_pct=1/100):
    '''
    ZigZag转折点
    
    TODO
    ----
    加入时间间隔控制

    Parameters
    ----------
    data : pd.DataFrame
        需包含[high_col, low_col]列
    high_col : str
        确定zigzag高点的数据列名
    low_col : str
        确定zigzag低点的数据列名
    up_pct, down_pct : float
        确定zigzag高/低转折点的幅度

    Returns
    -------
    zigzag : pd.Series
        返回zigzag标签序列。其中1/-1表示确定的高/低点；
        0.5/-0.5表示未达到偏离幅度而不能确定的高低点。
    '''

    def confirm_high(k):
        '''从前一个低点位置k开始确定下一个高点位置'''
        v0, pct_high, pct_high_low = df.loc[df.index[k], low_col], 0.0, 0.0
        Cmax, Cmax_idx = -np.inf, k+1
        k += 2
        while k < df.shape[0] and \
                              (pct_high_low > -down_pct or pct_high < up_pct):
            if df.loc[df.index[k-1], high_col] > Cmax:
                Cmax = df.loc[df.index[k-1], high_col]
                Cmax_idx = k-1
            pct_high = Cmax / v0 - 1

            pct_high_low = min(pct_high_low,
                                   df.loc[df.index[k], low_col] / Cmax - 1)

            k += 1

        if k == df.shape[0]:
            if df.loc[df.index[k-1], high_col] > Cmax:
                Cmax = df.loc[df.index[k-1], high_col]
                Cmax_idx = k-1
                pct_high = Cmax / v0 - 1
                pct_high_low = 0.0

        return Cmax_idx, pct_high >= up_pct, pct_high_low <= -down_pct

    def confirm_low(k):
        '''从前一个高点位置k开始确定下一个低点位置'''
        v0, pct_low, pct_low_high = df.loc[df.index[k], high_col], 0.0, 0.0
        Cmin, Cmin_idx = np.inf, k+1
        k += 2
        while k < df.shape[0] and \
                              (pct_low_high < up_pct or pct_low > -down_pct):
            if df.loc[df.index[k-1], low_col] < Cmin:
                Cmin = df.loc[df.index[k-1], low_col]
                Cmin_idx = k-1
            pct_low = Cmin / v0 - 1

            pct_low_high = max(pct_low_high,
                                   df.loc[df.index[k], high_col] / Cmin - 1)

            k += 1

        if k == df.shape[0]:
            if df.loc[df.index[k-1], low_col] < Cmin:
                Cmin = df.loc[df.index[k-1], low_col]
                Cmin_idx = k-1
                pct_low = Cmin / v0 - 1
                pct_low_high = 0.0

        return Cmin_idx, pct_low <= -down_pct, pct_low_high >= up_pct

    # 若data中已有zigzag列，先检查找出最后一个转折点已经能确定的位置，从此位置开始算
    if 'zigzag' in data.columns:
        cols = list(set([high_col, low_col])) + ['zigzag']
        df = data[cols].copy()
        k = df.shape[0] - 1
        while k > 0 and df.loc[df.index[k], 'zigzag'] in [0, 0.5, -0.5]:
            k -= 1
        ktype = df.loc[df.index[k], 'zigzag']
    # 若data中没有zigzag列或已有zigzag列不能确定有效的转折点，则从头开始算
    if 'zigzag' not in data.columns or ktype in [0, 0.5, -0.5]:
        cols = list(set([high_col, low_col]))
        df = data[cols].copy()
        df['zigzag'] = 0
        # 确定开始时的高/低点标签
        k1, OK_high, OK_high_low = confirm_high(0)
        OK1 = OK_high and OK_high_low
        k_1, OK_low, OK_low_high = confirm_low(0)
        OK_1 = OK_low and OK_low_high
        if not OK1 and not OK_1:
            return df['zigzag']
        elif OK1 and not OK_1:
            df.loc[df.index[0], 'zigzag'] = -0.5
            df.loc[df.index[k1], 'zigzag'] = 1
            k = k1
            ktype = 1
        elif not OK1 and OK_1:
            df.loc[df.index[0], 'zigzag'] = 0.5
            df.loc[df.index[k_1], 'zigzag'] = -1
            k = k_1
            ktype = -1
        elif k1 < k_1:
            df.loc[df.index[0], 'zigzag'] = -0.5
            df.loc[df.index[k1], 'zigzag'] = 1
            k = k1
            ktype = 1
        else:
            df.loc[df.index[0], 'zigzag'] = 0.5
            df.loc[df.index[k_1], 'zigzag'] = -1
            k = k_1
            ktype = -1

    while k < df.shape[0]:
        func_confirm = confirm_high if ktype == -1 else confirm_low
        k, OK_mid, OK_right = func_confirm(k)
        if OK_mid and OK_right:
            df.loc[df.index[k], 'zigzag'] = -ktype
            ktype = -ktype
        elif OK_mid:
            df.loc[df.index[k], 'zigzag'] = -ktype * 0.5
            break

    return df['zigzag']


def plot_candle_zz(data, zzcol='zigzag',
                   zz_high='high', zz_low='low',
                   **kwargs):
    '''在K线图上绘制ZigZag'''
    # data = data.copy()
    data['col_zz1'] = data[zzcol].apply(lambda x: 1 if x > 0 else 0)
    data['col_zz-1'] = data[zzcol].apply(lambda x: 1 if x < 0 else 0)
    data['col_zz'] = data['col_zz1'] * data[zz_high] + \
                     data['col_zz-1'] * data[zz_low]
    data['col_zz'] = data[['col_zz1', 'col_zz-1', 'col_zz']].apply(
                     lambda x: x['col_zz'] if x['col_zz1'] == 1 or \
                               x['col_zz-1'] == 1 else np.nan, axis=1)
    data['zz_loc'] = data['col_zz1'] + data['col_zz-1']
    if 'cols_to_label_info' in kwargs.keys():
        cols_to_label_info = kwargs['cols_to_label_info']
        del kwargs['cols_to_label_info']
        cols_to_label_info.update(
            {'col_zz': [['zz_loc', (1,), ('-b',), False]]})
    else:
        cols_to_label_info={'col_zz': [['zz_loc', (1,), ('-b',), False]]}
    if 'cols_other_upleft' in kwargs.keys():
        cols_other_upleft = kwargs['cols_other_upleft']
        del kwargs['cols_other_upleft']
        cols_other_upleft.update({'col_zz': ('.b', False)})
    else:
        cols_other_upleft = {'col_zz': ('.w', False)}
    plot_candle(data, cols_other_upleft=cols_other_upleft,
                cols_to_label_info=cols_to_label_info,
                **kwargs)


if __name__ == '__main__':
    from dramkit import load_csv

    # zigzig测试--------------------------------------------------------------
    fpath = '../test/510050_daily_pre_fq.csv'
    his_data = load_csv(fpath)
    his_data.rename(columns={'date': 'time'}, inplace=True)
    his_data.set_index('time', drop=False, inplace=True)

    # N = his_data.shape[0]
    N = 100
    col = 'close'
    data = his_data.iloc[-N:-1, :].copy()

    high_col, low_col, up_pct, down_pct = 'high', 'low', 3/100, 3/100
    data['zigzag'] = zigzag(data, high_col, low_col, up_pct, down_pct)
    plot_candle_zz(data, zz_high=high_col, zz_low=low_col,
                   args_ma=None, args_boll=None, plot_below=None,
                   grid=False, figsize=(10, 6))


    fpath = '../test/zigzag_test.csv'
    data = load_csv(fpath)
    dates = list(data['date'].unique())
    data = data[data['date'] == dates[0]].copy()
    plot_candle_zz(data, zzcol='label', args_ma=None,
                    args_boll=None, plot_below=None, figsize=(10, 6))

    high_col, low_col, up_pct, down_pct = 'high', 'low', 0.35/100, 0.35/100
    data['zigzag'] = zigzag(data, high_col, low_col, up_pct, down_pct)
    plot_candle_zz(data, args_ma=None, args_boll=[15, 2],
                   plot_below=None, figsize=(10, 6))
