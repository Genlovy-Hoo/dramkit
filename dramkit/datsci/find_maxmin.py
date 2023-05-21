# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from dramkit.logtools.utils_logger import logger_show
from dramkit.gentools import (con_count,
                              gap_count,
                              isnull,
                              cal_pct,
                              replace_repeat_pd,
                              get_preval_func_cond)
from dramkit.iotools import load_csv
from dramkit.datsci.preprocess import norm_linear
from dramkit.plottools.plot_common import plot_maxmins, plot_series

#%%
def find_maxmin(series, t_min=2, t_max=np.inf,
                min_dif_pct=None, min_dif_val=None,
                max_dif_pct=None, max_dif_val=None,
                pct_v00=1, skip_nan=True, logger=None):
    '''
    标注序列中的极值点
    
    TODO
    ----
    series中存在nan时的值填充及信号纠正处理

    Parameters
    ----------
    series: pd.Series
        待标注极值点的序列
    t_min : int
        设置极大极小值之间至少需要间隔t_min个点(相当于最小半周期)
    t_max : int
        设置极大极小值直接最大间隔期数(若间隔大于t_max，则直接保留极值点，不进行其它判断)
    min_dif_val, min_dif_pct : float
        若两个相邻极值点之间的间隔大于等于t_min，但是两者差值/百分比小于
        min_dif_val/min_dif_pct，则删除该极值点对
    max_dif_val, max_dif_pct : float
        若两个相邻极值点之间的间隔小于t_min，但是两者差值/百分比大于
        max_dif_val/max_dif_pct，则保留该极值点对（两个参数二选一）
    pct_v00 : float
        当使用百分比控制差值时，若前一个值为0时，指定的百分比变化值
    skip_nan : float
        当series中存在nan值时，是否继续，若为False，则抛出异常，为True则抛出警告信息
    logger : None, Logger
        日志记录器    

    Returns
    -------
    label : pd.Series
        标注的极值序列，其值中1表示极大值点，-1表示极小值点，0表示普通点
    '''

    if len(series) < 2:
        raise ValueError('输入series长度不能小于2！')

    if not isnull(min_dif_pct) and not isnull(min_dif_val):
        logger_show('同时设置`min_dif_pct`和`min_dif_val`，以`min_dif_pct`为准！',
                    logger, 'warn')
        min_dif_val = None
    if not isnull(max_dif_pct) and not isnull(max_dif_val):
        logger_show('同时设置`max_dif_pct`和`max_dif_val`，以`max_dif_pct`为准！',
                    logger, 'warn')
        max_dif_val = None

    series = pd.Series(series)

    if series.isna().sum() > 0:        
        if skip_nan:
            logger_show('发现无效值，可能导致结果错误！', logger, 'warn')
        else:
            raise ValueError('发现无效值，请先处理！')

    # 序列名和索引名
    if series.name is None:
        series.name = 'series'
    if series.index.name is None:
        series.index.name = 'idx'

    df = pd.DataFrame(series)
    col = df.columns[0]
    df.reset_index(inplace=True)
    
    # plot_series(df, {col: '.-k'}, title='原始数据', grids=True)

    # 所有转折点
    df['dif'] = df[col].diff()
    df['dif_big'] = (df['dif'] > 0).astype(int)
    df['dif_sml'] = -1 * (df['dif'] < 0).astype(int)
    ktmp = 1
    while ktmp < df.shape[0] and df.loc[df.index[ktmp], 'dif'] == 0:
        ktmp += 1
    if df.loc[df.index[ktmp], 'dif'] > 0:
        df.loc[df.index[0], 'dif_sml'] = -1
    elif df.loc[df.index[ktmp], 'dif'] < 0:
        df.loc[df.index[0], 'dif_big'] = 1
    df['label'] = df['dif_big'] + df['dif_sml']
    label_ = replace_repeat_pd(df['label'][::-1], 1, 0)
    label_ = replace_repeat_pd(label_, -1, 0)
    df['label'] = label_[::-1]

    # plot_maxmins(df, col, 'label', title='所有拐点')

    # t_min应大于等于1
    if t_min is not None and t_min < 1:
        t_min = None

    if t_min:
        def _max_dif_ok(v2, v3):
            '''判断v2和v3差值是否大于max_dif_pct/max_dif_val'''
            if not isnull(max_dif_pct) and \
                                abs(cal_pct(v2, v3, pct_v00)) >= max_dif_pct:
                return True
            elif not isnull(max_dif_val) and abs(v3-v2) >= max_dif_val:
                return True
            else:
                return False

        def _min_dif_ok(v2, v3):
            '''判断v2和v3差值是否小于min_dif_pct/min_dif_val'''
            if not isnull(min_dif_pct) and \
                                abs(cal_pct(v2, v3, pct_v00)) <= min_dif_pct:
                return True
            elif not isnull(min_dif_val) and abs(v3-v2) <= min_dif_val:
                return True
            else:
                return False

        def _to_del(df, k1, k2, k3, k4):
            '''
            判断是否需要删除(k2, k3)位置的极值点对，判断条件见_del_t_min函数
            '''
            if k4 < df.shape[0] and k1 < 0: # 开头位置特殊处理
                v2, v3, v4 = df.loc[k2, col], df.loc[k3, col], df.loc[k4, col]
                if k3-k2 < t_min+1: # 间隔小于t_min
                    if df.loc[k2, 'label'] == 1:
                        if v2 >= v4: # 若影响前后趋势，保留
                            return False, None
                        elif _max_dif_ok(v2, v3): # 差值满足条件，保留
                            return False, None
                        else:
                            return True, k2
                    elif df.loc[k2, 'label'] == -1:
                        if v2 <= v4: # 若影响前后趋势，保留
                            return False, None
                        elif _max_dif_ok(v2, v3): # 差值满足条件，保留
                            return False, None
                        else:
                            return True, k2
                elif k3-k2 >= t_max: # 间隔大于t_max
                    return False, None
                else: # 间隔大于等于t_min
                    if df.loc[k2, 'label'] == 1:
                        if v2 >= v4: # 若影响前后趋势，保留
                            return False, None
                        elif _min_dif_ok(v2, v3): # 不满足差值条件，删除
                            return True, k2
                        else:
                            return False, None
                    elif df.loc[k2, 'label'] == -1:
                        if v2 <= v4: # 若影响前后趋势，保留
                            return False, None
                        elif _min_dif_ok(v2, v3): # 不满足差值条件，删除
                            return True, k2
                        else:
                            return False, None
            elif k4 < df.shape[0] and k1 > -1:
                v2, v3 = df.loc[k2, col], df.loc[k3, col]
                v1, v4 = df.loc[k1, col], df.loc[k4, col]
                if k3-k2 < t_min+1: # 间隔小于t_min
                    if df.loc[k2, 'label'] == 1:
                        if v2 > v4 or v3 < v1: # 若影响前后趋势，保留
                            return False, None
                        elif _max_dif_ok(v2, v3): # 差值满足条件，保留
                            return False, None
                        else:
                            return True, [k2, k3]
                    elif df.loc[k2, 'label'] == -1:
                        if v2 < v4 or v3 > v1: # 若影响前后趋势，保留
                            return False, None
                        elif _max_dif_ok(v2, v3): # 差值满足条件，保留
                            return False, None
                        else:
                            return True, [k2, k3]
                elif k3-k2 >= t_max: # 间隔大于t_max
                    return False, None
                else: # 间隔大于等于t_min
                    if df.loc[k2, 'label'] == 1:
                        if v2 > v4 or v3 < v1: # 若影响前后趋势，保留
                            return False, None
                        elif _min_dif_ok(v2, v3): # 不满足差值条件，删除
                            return True, [k2, k3]
                        else:
                            return False, None
                    elif df.loc[k2, 'label'] == -1:
                        if v2 < v4 or v3 > v1: # 若影响前后趋势，保留
                            return False, None
                        elif _min_dif_ok(v2, v3): # 不满足差值条件，删除
                            return True, [k2, k3]
                        else:
                            return False, None
            else:
                return False, None

        def _del_t_min(df):
            '''
            | 删除不满足条件的相邻极大极小值点对，删除条件：
            |     若间隔小于t_min：
            |         若删除后影响前后趋势，保留
            |         若差值大于max_dif_pct/max_dif_val，保留
            |         否则删除
            |     若间隔大于等于t_min：
            |         若差值小于min_dif_pct/min_dif_val，删除
            |         否则保留
            | 注：df中数据的排序依据为df.index
            '''

            k2 = 0
            while k2 < df.shape[0]:
                if df.loc[k2, 'label'] == 0:
                    k2 += 1
                else:
                    k1 = k2-1
                    while k1 > -1 and df.loc[k1, 'label'] == 0:
                        k1 -= 1

                    k3 = k2+1
                    while k3 < df.shape[0] and df.loc[k3, 'label'] == 0:
                        k3 += 1

                    k4 = k3 +1
                    while k4 < df.shape[0] and df.loc[k4, 'label'] == 0:
                        k4 += 1

                    to_del, idxs = _to_del(df, k1, k2, k3, k4)
                    if to_del:
                        df.loc[idxs, 'label'] = 0
                        # plot_maxmins(df, col, 'label', title='筛除不满足条件的极值对', grid=False)

                    k2 = k3

            return df

        df = _del_t_min(df)
        # plot_maxmins(df, col, 'label', title='第1次正向筛除')

        df.index = range(df.shape[0]-1, -1, -1)
        df = _del_t_min(df)
        
        def _check_t_min(df, t_min):
            Fcond = lambda x: True if x == 0 else False
            df['tmp'] = con_count(df['label'], Fcond).shift(1)
            df['tmp'] = abs(df['tmp'] * df['label'])
            df.loc[df.index[0], 'tmp'] = 0
            tmp = list(df[df['label'] != 0]['tmp'])
            df.drop('tmp', axis=1, inplace=True)
            if len(tmp) <= 3:
                return True, tmp
            else:
                tmp = tmp[1:]
                if all([x >= t_min for x in tmp]):
                    return True, tmp
                else:
                    return False, tmp
        t_min_ok, tmp = _check_t_min(df, t_min)
        tmp_new = []
        # plot_maxmins(df, col, 'label', title='第1次反向筛除，t_min check: '+str(t_min_ok))
        # 注：特殊情况下不可能满足任何两个极大极小值对之间的间隔都大于t_min
        k = 2
        while not t_min_ok and not tmp == tmp_new:
            t_min_ok, tmp = _check_t_min(df, t_min)
            df.index = range(df.shape[0])
            df = _del_t_min(df)
            # plot_maxmins(df, col, 'label', title='第%s次正向筛除'%k)
            df.index = range(df.shape[0]-1, -1, -1)
            df = _del_t_min(df)
            t_min_ok, tmp_new = _check_t_min(df, t_min)
            # plot_maxmins(df, col, 'label', title='第%s次反向筛除，t_min check: %s'%(k, t_min_ok))
            k += 1

    df.set_index(series.index.name, inplace=True)

    return df['label']

#%%
def check_maxmins(df, col, col_label, max_lbl=1, min_lbl=-1):
    '''
    | 检查df中col_label指定列的极值点排列是否正确
    | 要求df须包含指定的两列，其中：
    | - col_label指定列保存极值点，其值max_lbl(int，默认1)表示极大值，
        min_lbl(int，默认-1)表示极小值，其余(默认0)为普通点
    | - col列为序列数值列
    |
    | 返回判断结果(True或False)以及错误信息
    '''

    tmp = df[[col, col_label]].reset_index()
    df_part = tmp[tmp[col_label].isin([max_lbl, min_lbl])]

    if df_part.shape[0] == 0:
        return False, '没有发现极值点，检查输入参数！'
    if df_part.shape[0] == 1:
        if df_part[col].iloc[0] in [df[col].max(), df[col].min()]:
            return True, '只发现1个极值点！'
        else:
            return False, '只发现一个极值点且不是最大或最小值！'
    if df_part.shape[0] == 2:
        vMax = df_part[df_part[col_label] == max_lbl][col].iloc[0]
        vMin = df_part[df_part[col_label] == min_lbl][col].iloc[0]
        if vMax <= vMin:
            return False, '只发现两个极值点且极大值小于等于极小值！'
        if vMax != df[col].max():
            return False, '只发现两个极值点且极大值不是最大值！'
        if vMin != df[col].min():
            return False, '只发现两个极值点且极小值不是最小值！'
        return True, '只发现两个极值点！'

    # 不能出现连续的极大/极小值点
    label_diff = list(df_part[col_label].diff().unique())
    if 0 in label_diff:
        return False, '存在连续极大/极小值点！'

    # 极大/小值点必须大/小于极小/大值点
    for k in range(1, df_part.shape[0]-1):
        if df_part[col_label].iloc[k] == max_lbl:
            if df_part[col].iloc[k] <= df_part[col].iloc[k-1] or \
                            df_part[col].iloc[k] <= df_part[col].iloc[k+1]:
                return False, ('极大值点小于等于极小值点！',
                               df.index[df_part.index[k]])
        else:
            if df_part[col].iloc[k] >= df_part[col].iloc[k-1] or \
                            df_part[col].iloc[k] >= df_part[col].iloc[k+1]:
                return False, ('极小值点大于等于极大值点！',
                               df.index[df_part.index[k]])

    # 极大极小值点必须是闭区间内的最大最小值
    for k in range(0, df_part.shape[0]-1):
        idx1 = df_part.index[k]
        idx2 = df_part.index[k+1]
        if tmp.loc[idx1, col_label] == max_lbl:
            if tmp.loc[idx1, col] != tmp.loc[idx1:idx2, col].max() or \
                    tmp.loc[idx2, col] != tmp.loc[idx1:idx2, col].min():
                return False, ('极大极小值不是闭区间内的最大最小值！',
                               [df.index[idx1], df.index[idx2]])
        else:
            if tmp.loc[idx1, col] != tmp.loc[idx1:idx2, col].min() or \
                    tmp.loc[idx2, col] != tmp.loc[idx1:idx2, col].max():
                return False, ('极大极小值不是闭区间内的最大最小值！',
                               [df.index[idx1], df.index[idx2]])

    # 开头和结尾部分单独判断
    if df_part.index[0] != 0:
        first_loc = df_part.index[0]
        if df_part[col_label].iloc[0] == -1:
            min_ = df[col].iloc[:first_loc].min()
            if df[col].iloc[first_loc] > min_:
                return False, ('第一个极小值点错误！', df.index[first_loc])
        elif df[col].iloc[first_loc] == 1:
            max_ = df[col].iloc[:first_loc].max()
            if df[col].iloc[first_loc] < max_:
                return False, ('第一个极大值点错误！', df.index[first_loc])
    if df_part.index[-1] != df.shape[0]-1:
        last_loc = df_part.index[-1]
        if df_part[col_label].iloc[-1] == -1:
            min_ = df[col].iloc[last_loc+1:].min()
            if df[col].iloc[last_loc] > min_:
                return False, ('最后一个极小值点错误！', df.index[last_loc])
        elif df_part[col_label].iloc[-1] == 1:
            max_ = df[col].iloc[last_loc+1:].max()
            if df[col].iloc[last_loc] < max_:
                return False, ('最后一个极大值点错误！', df.index[last_loc])

    return True, None

#%%
def find_maxmin_rolling(series, window, cal_n=None, **kwargs):
    '''
    滚动寻找极值点
    
    Examples
    --------
    >>> fpath = '../_test/510500.SH_daily_qfq.csv'
    >>> df = load_csv(fpath)
    >>> df = df.set_index('date', drop=False).iloc[-200:, :]
    >>> window, t_min = 100, 3
    >>> col = 'close'
    >>> df['label'] = find_maxmin_rolling(df[col],
    ...                                   window=window,
    ...                                   cal_n=50,
    ...                                   t_min=t_min)
    >>> plot_maxmins(df.iloc[:, :], col, 'label', figsize=(12, 7))
    '''
    series = pd.Series(series)
    n = len(series)
    res = pd.Series([np.nan] * n, index=series.index, name='label')
    tqdm.write('find maxmin rolling...')
    time.sleep(0.5)
    start = window if isnull(cal_n) else max(window, n-cal_n)
    for k in tqdm(range(start, n+1)):
        subseries = series.iloc[k-window:k]
        res.iloc[k-1] = find_maxmin(subseries, **kwargs).iloc[-1]
    return res


def find_maxmin_cum(series, window_min=None, cal_n=None, **kwargs):
    '''
    滚动寻找极值点（用累计历史数据）
    
    Examples
    --------
    >>> fpath = '../_test/510500.SH_daily_qfq.csv'
    >>> df = load_csv(fpath)
    >>> df = df.set_index('date', drop=False).iloc[-200:, :]
    >>> window_min, t_min = 100, 3
    >>> col = 'close'
    >>> df['label'] = find_maxmin_cum(df[col],
    ...                               window_min=window_min,
    ...                               cal_n=50,
    ...                               t_min=t_min)
    >>> plot_maxmins(df.iloc[:, :], col, 'label', figsize=(12, 7))
    '''
    window_min = 1 if isnull(window_min) else window_min
    series = pd.Series(series)
    n = len(series)
    res = pd.Series([np.nan] * n, index=series.index, name='label')
    tqdm.write('find maxmin cum...')
    time.sleep(0.5)
    start = window_min if isnull(cal_n) else max(window_min, n-cal_n)
    for k in tqdm(range(start, n+1)):
        subseries = series.iloc[:k]
        res.iloc[k-1] = find_maxmin(subseries, **kwargs).iloc[-1]
    return res

#%%
def get_his_maxmin_info(df, col, col_label, max_lbl=1, min_lbl=-1):
    '''获取历史极值点序列信息'''
    df = df.reindex(columns=[col, col_label])
    df['pre_label_val'] = get_preval_func_cond(df, col, col_label,
                                               lambda x: x in [max_lbl, min_lbl])
    df['dif_pre_label_val'] = df[col] - df['pre_label_val']
    df['pct_pre_label_val'] = df[col] / df['pre_label_val'] - 1
    df['gap_pre_label'] = gap_count(df[col_label],
                                    lambda x: x in [max_lbl, min_lbl])
    df.dropna(how='any', inplace=True)
    df = df[df[col_label].isin([max_lbl, min_lbl])]
    his_info = {}
    his_info['min2max'] = \
                {'difs': df[df[col_label] == max_lbl]['dif_pre_label_val'],
                 'pcts': df[df[col_label] == max_lbl]['pct_pre_label_val'],
                 'gaps': df[df[col_label] == max_lbl]['gap_pre_label']}
    his_info['max2min'] = \
                {'difs': df[df[col_label] == min_lbl]['dif_pre_label_val'],
                 'pcts': df[df[col_label] == min_lbl]['pct_pre_label_val'],
                 'gaps': df[df[col_label] == min_lbl]['gap_pre_label']}
    return his_info, df


def get_last_pos_info(df, col, col_label, min_gap=2, max_lbl=1, min_lbl=-1):
    '''
    | 获取最后一条记录在极大极小值间的位置信息
    | min_gap若大于0，则强制将最后min_gap条记录的标签去除
    '''
    df = df.reindex(columns=[col, col_label])
    if min_gap > 0:
        df.loc[df.index[-min_gap:], col_label] = 0
    df['pre_max_val'] = get_preval_func_cond(df, col, col_label,
                                             lambda x: x == max_lbl)
    df['pct_pre_max_val'] = df['pre_max_val'] / df[col] - 1
    df['dif_pre_max_val'] = df['pre_max_val'] - df[col]
    df['pre_min_val'] = get_preval_func_cond(df, col, col_label,
                                             lambda x: x == min_lbl)
    df['pct_pre_min_val'] = df[col] / df['pre_min_val'] - 1
    df['dif_pre_min_val'] = df[col] - df['pre_min_val']
    df['gap_pre_max'] = gap_count(df[col_label], lambda x: x == max_lbl)
    df['gap_pre_min'] = gap_count(df[col_label], lambda x: x == min_lbl)
    # df.dropna(how='any', inplace=True)
    his_pcts_min2max = df[df[col_label] == max_lbl]['pct_pre_min_val']
    his_difs_min2max = df[df[col_label] == max_lbl]['dif_pre_min_val']
    his_gaps_min2max = df[df[col_label] == max_lbl]['gap_pre_min']
    his_pcts_max2min = df[df[col_label] == min_lbl]['pct_pre_max_val']
    his_difs_max2min = df[df[col_label] == min_lbl]['dif_pre_max_val']
    his_gaps_max2min = df[df[col_label] == min_lbl]['gap_pre_max']
    pct_min2now = df.loc[df.index[-1], 'pct_pre_min_val']
    dif_min2now = df.loc[df.index[-1], 'dif_pre_min_val']
    gap_min2now = df.loc[df.index[-1], 'gap_pre_min']
    pct_max2now = df.loc[df.index[-1], 'pct_pre_max_val']
    dif_max2now = df.loc[df.index[-1], 'dif_pre_max_val']
    gap_max2now = df.loc[df.index[-1], 'gap_pre_max']
    pos_pct_min2now = norm_linear(pct_min2now, his_pcts_min2max.min(),
                                  his_pcts_min2max.max(), x_must_in_range=False)
    pos_dif_min2now = norm_linear(dif_min2now, his_difs_min2max.min(),
                                  his_difs_min2max.max(), x_must_in_range=False)
    pos_gap_min2now = norm_linear(gap_min2now, his_gaps_min2max.min(),
                                  his_gaps_min2max.max(), x_must_in_range=False)
    pos_pct_max2now = norm_linear(pct_max2now, his_pcts_max2min.min(),
                                  his_pcts_max2min.max(), x_must_in_range=False,
                                  reverse=False)
    pos_dif_max2now = norm_linear(dif_max2now, his_difs_max2min.min(),
                                  his_difs_max2min.max(), x_must_in_range=False,
                                  reverse=False)
    pos_gap_max2now = norm_linear(gap_max2now, his_gaps_max2min.min(),
                                  his_gaps_max2min.max(), x_must_in_range=False)
    tmp = df[df[col_label].isin([max_lbl, min_lbl])]
    last_label = tmp[col_label].iloc[-1]
    pre_label = last_label
    if df[col_label].iloc[-1] in [max_lbl, min_lbl]:
        pre_label = tmp[col_label].iloc[-2]
    return {'pos_pct_min2now': pos_pct_min2now,
            'pos_dif_min2now': pos_dif_min2now,
            'pos_gap_min2now': pos_gap_min2now,
            'pos_pct_max2now': pos_pct_max2now,
            'pos_dif_max2now': pos_dif_max2now,
            'pos_gap_max2now': pos_gap_max2now,
            'pct_min2now': pct_min2now,
            'dif_min2now': dif_min2now,
            'gap_min2now': gap_min2now,
            'pct_max2now': pct_max2now,
            'dif_max2now': dif_max2now,
            'gap_max2now': gap_max2now,
            'pre_max_val': df['pre_max_val'].iloc[-1],
            'pre_min_val': df['pre_min_val'].iloc[-1],
            'last_label': last_label,
            'pre_label': pre_label}

#%%
def get_maxmin_neighbor_label(df, label_col, base_col, labels=[-1, 1],
                              label0=0, max_t=3, max_pct=1.0/100):
    '''
    极大极小值邻近点标签标注
    
    TODO
    ----
    处理不同标签重复标注问题
    '''
    df = df.reindex(columns=[base_col, label_col])
    df[label_col+'_new'] = label0
    df['idx'] = range(0, df.shape[0])
    imax = df['idx'].max()
    for lbl in labels:
        idxs = list(df[df[label_col] == lbl]['idx'])
        for idx in idxs:
            v0 = df[base_col].iloc[idx]
            strt = max(0, idx-max_t)
            end = min(idx+max_t, imax)
            for k in range(strt, end+1):
                if abs(df[base_col].iloc[k] / v0 - 1) < max_pct:
                    df.loc[df.index[k], label_col+'_new'] = lbl
    return df[label_col+'_new']

#%%
def del_tooclose_maxmin(df, label_col, base_col, min_pct=5.0/100):
    '''
    删除 :func:`find_maxmin` 结果中极大值和极小值差别太小的极值点
    '''
    df = df.reindex(columns=[base_col, label_col])
    k, k1 = 0, 0
    while k < df.shape[0]-1 and k1 < df.shape[0]-1:
        if df.loc[df.index[k], label_col] == 0:
            k += 1
        else:
            k1 = k + 1
            while k1 < df.shape[0]-1 and df.loc[df.index[k1], label_col] == 0:
                k1 += 1
            if df.loc[df.index[k1], label_col] != 0:
                v0 = df.loc[df.index[k], base_col]
                v1 = df.loc[df.index[k1], base_col]
                if abs(v0 / v1 - 1) < min_pct:
                    df.loc[df.index[[k, k1]], label_col] = 0
            k = k1
    return df[label_col]

#%%
def get_maxmin_records(df, maxmin_col='maxmin'):
    '''获取所有maxmin记录'''
    maxmin_records = df.copy()
    maxmin_records['idx'] = range(maxmin_records.shape[0])
    maxmin_records = maxmin_records[maxmin_records[maxmin_col] != 0].copy()
    return maxmin_records


def get_last_sure_maxmin_info(df, maxmin_col='maxmin', price_col='ma',
                              sure_gap_min=5, sure_pct_min=0.1/100):
    '''获取最后一个确定的maxmin标签信息'''
    maxmin_records = get_maxmin_records(df, maxmin_col=maxmin_col)
    last_is_maxmin = False # 最后一条记录maxmin是否不为0
    if maxmin_records['idx'].iloc[-1] == df.shape[0]-1:
        maxmin_records = maxmin_records.iloc[:-1, :]
        last_is_maxmin = True
    sure_loc = maxmin_records['idx'].iloc[-1] # maxmin位置
    sure_maxmin = maxmin_records[maxmin_col].iloc[-1] # maxmin标签值
    sure_gap = df.shape[0]-1 - sure_loc # 时间间隔
    sure_price = maxmin_records[price_col].iloc[-1] # maxmin价位
    sure_pct = df[price_col].iloc[-1] / sure_price - 1 # 价差百分比
    if sure_gap > sure_gap_min and abs(sure_pct) > sure_pct_min:
        sure = True
    else:
        sure = False
    other_info = (sure_maxmin, sure_loc, sure_gap, sure_price, sure_pct)
    return sure, last_is_maxmin, other_info
            


def get_continue_maxmin_info(maxmins_his, maxmin_col='maxmin'):
    '''获取连续maxmin标签的连续期数和信号类型'''
    idx_maxmin = [(x['idx'].iloc[-1], x[maxmin_col].iloc[-1]) \
                  for x in maxmins_his]
    idx_last, maxmin_last = idx_maxmin[-1][0], idx_maxmin[-1][1]
    itmp = len(idx_maxmin)-2
    n_con = 1
    while itmp > 0:
        idx, maxmin = idx_maxmin[itmp][0], idx_maxmin[itmp][1]
        if idx == idx_last - 1 and maxmin == maxmin_last:
            n_con += 1
            idx_last = idx
            itmp -= 1
        else:
            break
    return n_con, maxmin_last


def disaper(maxmins_his, loc, col_iloc='idx', tol_gap=10):
    '''
    | 判断maxmin信号消失
    | loc指定位置，tol_gap设置误差范围
    '''
    if loc == 0:
        return False
    idxs_his = [list(x[col_iloc]) for x in maxmins_his]
    def has_in_range(idxs, loc):
        return any([loc-tol_gap <= x <= loc+tol_gap for x in idxs])
    include_loc = [has_in_range(idxs, loc) for idxs in idxs_his]
    idx_strt = min([k for k in range(len(include_loc)) if include_loc[k]])
    return not all([has_in_range(idxs, loc) for idxs in idxs_his[idx_strt:]])

#%%
def find_maxmin_dy(df, col, t_min, his_lag=None, use_all=True,
        minGap_min2now=0, minDif_min2now=None, minPct_min2now=0,
        maxGap_min2now=np.inf, maxDif_min2now=None, maxPct_min2now=np.inf,
        dirtSureMinGap=np.inf, dirtSureMinDif=None, dirtSureMinPct=np.inf,
        minGap_max2nowMin=0, minDif_max2nowMin=None, minPct_max2nowMin=0,
        dirtSureMinGap_max2now=np.inf, dirtSureMinDif_max2now=None,
        dirtSureMinPct_max2now=np.inf,
        Min_max_now=np.inf, Min_max_pre_min=np.inf,
        minGap_max2now=0, minDif_max2now=None, minPct_max2now=0,
        maxGap_max2now=np.inf, maxDif_max2now=None, maxPct_max2now=np.inf,
        dirtSureMaxGap=np.inf, dirtSureMaxDif=None, dirtSureMaxPct=np.inf,
        minGap_min2nowMax=0, minDif_min2nowMax=None, minPct_min2nowMax=0,
        dirtSureMaxGap_min2now=np.inf, dirtSureMaxDif_min2now=None,
        dirtSureMaxPct_min2now=np.inf,
        Max_min_now=-np.inf, Max_min_pre_max=-np.inf,
        logger=None, n_log=None, fig_save_dir=None, plot_all=False,
        plot_lag=None, kwargs_findMaxMin={}, kwargs_plot={}, plot_sleep=0):
    '''
    动态标签确定
    
    TODO
    ----
    待重新实现

    Parameters
    ----------
    his_lag : int, None
        从第his_lag条记录处开始动态标注，若为None则默认为4*t_min
    use_all : bool
        若为True，每次动态标注时使用之前的所有历史数据，否则只使用前his_lag期数据
    动态确定标签参数和规则（以确定极小值信号为例，极大值同理），（待补充） : 
    logger : Logger, None
        日志记录器
    n_log : int, None
        每隔n_log期日志提示进度
    fig_save_dir : str, bool
        若为False，则不做图；若为文件夹路径，则画图并保存到文件夹；若为None，则绘图但不保存
    plot_all : bool
        若为False，则只有在确定为极大/小值是才绘图，否则每期都绘图
    plot_lag : None, str, int
        指定每次画图时往前取plot_lag期历史数据进行绘图。
        若为None，则取his_lag；若为`all`，则取所有历史期数。
    '''

    kwargs_findMaxMin.update({'logger': logger})

    data = df.copy()
    it_idxs = list(data.index)
    N = len(it_idxs)
    # 开始历史期数
    his_lag = 4 * t_min if his_lag is None else his_lag

    data['signal'] = 0 # 动态标签信号
    data['sig_type'] = '' # 信号类型，包括：正常、消失反向、遗漏追加
    last_sig = np.nan # 前一个信号
    last_sig_type = np.nan # 前一个信号类型
    last_sig_mm_loc = np.nan # 前一个信号对应极值点位置

    data['iloc'] = range(0, data.shape[0])
    maxmins_his = []
    for it in range(his_lag, N):
        it_idx = data.index[it]
        if n_log and it % n_log == 0:
            logger_show('{} / {}, {} find_maxmin_dy ...'.format(it, N, it_idx),
                        logger, 'info')

        if use_all:
            df = data.loc[data.index[:it+1], :].copy()
        else:
            df = data.loc[data.index[it-his_lag:it+1], :].copy()

        # 极值点
        df['mm'] = find_maxmin(df[col], t_min=t_min, **kwargs_findMaxMin)
        maxmins = get_maxmin_records(df, maxmin_col='mm')
        maxmins_his.append(maxmins)

        # 最后一条记录的周期位置信息
        last_info = get_last_pos_info(df, col, 'mm', min_gap=0)
        # print(it_idx)
        # pprint(last_info)

        signal = 0
        sig_type = ''

        # 极值点消失
        if '正常' in str(last_sig_type):
            if disaper(maxmins_his, last_sig_mm_loc,
                                                  col_iloc='iloc', tol_gap=0):
                signal = -1 if 'Max' in last_sig_type else 1
                sig_type = '_消失反向Max' if signal == 1 else '_消失反向Min'
                last_sig_mm_loc = \
                    df[df['mm'] == last_info['pre_label']]['iloc'].iloc[-1]

        # 正常信号
        if df['mm'].iloc[-1] in [1, -1]: # 拐点未到，不用判断
            pass
        elif last_info['pre_label'] == last_sig: # 避免重复判断
            pass
        else:
            if last_info['pre_label'] == -1:
                # 条件1：距离上一个极小值点期数在给定范围内
                cond1 = minGap_min2now < \
                                    last_info['gap_min2now'] < maxGap_min2now

                # 条件2：距离上一个极小值点差值|百分比幅度在给定范围内
                if isnull(minDif_min2now) and isnull(maxDif_min2now):
                    cond2dif = True
                else:
                    cond2dif = minDif_min2now < \
                                    last_info['dif_min2now'] < maxDif_min2now
                if isnull(minPct_min2now) and isnull(maxPct_min2now):
                    cond2pct = True
                else:
                    cond2pct = minPct_min2now < \
                                    last_info['pct_min2now'] < maxPct_min2now
                cond2 = cond2dif and cond2pct

                cond12 = cond1 and cond2

                # 条件3：（直接判断）距离上一个极小值点期数超过给定阈值
                cond3 = last_info['gap_min2now'] > dirtSureMinGap

                # 条件4：距离上一个极小值点差值|百分比幅度超过给定阈值
                if isnull(dirtSureMinDif):
                    cond4dif = True
                else:
                    cond4dif = last_info['dif_min2now'] > dirtSureMinDif
                if isnull(dirtSureMinPct):
                    cond4pct = True
                else:
                    cond4pct = last_info['pct_min2now'] > dirtSureMinPct
                cond4 = cond4dif and cond4pct

                cond34 = cond3 or cond4

                # 条件5：距离上一个极小值点期数小于距离上一个极大值点期数
                cond5 = last_info['gap_min2now'] < last_info['gap_max2now']

                # 条件6：距离上一个极大值点期数超过给定阈值
                cond6 = last_info['gap_max2now'] > minGap_max2nowMin

                # 条件7：距离上一个极大值点差值|百分比幅度超过给定阈值
                if isnull(minDif_max2nowMin):
                    cond7dif = True
                else:
                    cond7dif = last_info['dif_max2now'] > minDif_max2nowMin
                if isnull(minPct_max2nowMin):
                    cond7pct = True
                else:
                    cond7pct = last_info['pct_max2now'] > minPct_max2nowMin
                cond7 = cond7dif and cond7pct

                cond67 = cond6 and cond7

                # 条件8：（直接判断）距离上一个极大值点期数超过给定阈值
                cond8 = last_info['gap_max2now'] > dirtSureMinGap_max2now

                # 条件9：（直接判断）距离上一个极大值点差值|百分比幅度超过给定阈值
                if isnull(dirtSureMinDif_max2now):
                    cond9dif = True
                else:
                    cond9dif = last_info['dif_max2now'] > \
                                                        dirtSureMinDif_max2now
                if isnull(dirtSureMinPct_max2now):
                    cond9pct = True
                else:
                    cond9pct = last_info['pct_max2now'] > \
                                                        dirtSureMinPct_max2now
                cond9 = cond9dif and cond9pct

                cond89 = cond8 or cond9

                # 条件10：当前col列值须小于给定阈值
                cond10 = df[col].iloc[-1] < Min_max_now

                # 条件11：前一个极小值点须小于给定阈值
                cond11 = last_info['pre_min_val'] < Min_max_pre_min

                cond1011 = cond10 and cond11

                # 信号确认
                if cond5 and (cond12 or cond34) and (cond67 or cond89) \
                                                and cond1011:
                    signal = -1
                    sig_type = '_正常Min'
                    last_sig_mm_loc = df[df['mm'] != 0]['iloc'].iloc[-1]

            elif last_info['pre_label'] == 1:
                # 条件1：距离上一个极大值点期数在给定范围内
                cond1 = minGap_max2now < \
                                    last_info['gap_max2now'] < maxGap_max2now

                # 条件2：距离上一个极大值点差值|百分比幅度在给定范围内
                if isnull(minDif_max2now) and isnull(maxDif_max2now):
                    cond2dif = True
                else:
                    cond2dif = minDif_max2now < \
                                    last_info['dif_max2now'] < maxDif_max2now
                if isnull(minPct_max2now) and isnull(maxPct_max2now):
                    cond2pct = True
                else:
                    cond2pct = minPct_max2now < \
                                    last_info['pct_max2now'] < maxPct_max2now
                cond2 = cond2dif and cond2pct

                cond12 = cond1 and cond2

                # 条件3：（直接判断）距离上一个极大值点期数超过给定阈值
                cond3 = last_info['gap_max2now'] > dirtSureMaxGap

                # 条件4：距离上一个极大值点差值|百分比幅度超过给定阈值
                if isnull(dirtSureMaxDif):
                    cond4dif = True
                else:
                    cond4dif = last_info['dif_max2now'] > dirtSureMaxDif
                if isnull(dirtSureMaxPct):
                    cond4pct = True
                else:
                    cond4pct = last_info['pct_max2now'] > dirtSureMaxPct
                cond4 = cond4dif and cond4pct

                cond34 = cond3 or cond4

                # 条件5：距离上一个极大值点期数小于距离上一个极小值点期数
                cond5 = last_info['gap_max2now'] < last_info['gap_min2now']

                # 条件6：距离上一个极小值点期数超过给定阈值
                cond6 = last_info['gap_min2now'] > minGap_min2nowMax

                # 条件7：距离上一个极小值点差值|百分比幅度超过给定阈值
                if isnull(minDif_min2nowMax):
                    cond7dif = True
                else:
                    cond7dif = last_info['dif_min2now'] > minDif_min2nowMax
                if isnull(minPct_min2nowMax):
                    cond7pct = True
                else:
                    cond7pct = last_info['pct_min2now'] > minPct_min2nowMax
                cond7 = cond7dif and cond7pct

                cond67 = cond6 and cond7

                # 条件8：（直接判断）距离上一个极小值点期数超过给定阈值
                cond8 = last_info['gap_min2now'] > dirtSureMaxGap_min2now

                # 条件9：（直接判断）距离上一个极小值点差值|百分比幅度超过给定阈值
                if isnull(dirtSureMaxDif_min2now):
                    cond9dif = True
                else:
                    cond9dif = last_info['dif_min2now'] > \
                                                        dirtSureMaxDif_min2now
                if isnull(dirtSureMaxPct_min2now):
                    cond9pct = True
                else:
                    cond9pct = last_info['pct_min2now'] > \
                                                        dirtSureMaxPct_min2now
                cond9 = cond9dif and cond9pct

                cond89 = cond8 or cond9

                # 条件10：当前col列值须大于给定阈值
                cond10 = df[col].iloc[-1] > Max_min_now

                # 条件11：前一个极大值点须大于给定阈值
                cond11 = last_info['pre_max_val'] > Max_min_pre_max

                cond1011 = cond10 and cond11

                # 信号确认
                if cond5 and (cond12 or cond34) and (cond67 or cond89) \
                                                and cond1011:
                    signal = 1
                    sig_type = '_正常Max'
                    last_sig_mm_loc = df[df['mm'] != 0]['iloc'].iloc[-1]

        # 极值点遗漏
        if last_sig == -1 and last_info['pre_label'] == 1 and signal == 0:
            if df[col].iloc[-1] < df[col].iloc[last_sig_mm_loc]:
                signal = 1
                sig_type = '_正常Max'
                last_sig_mm_loc = \
                    df[df['mm'] == last_info['pre_label']]['iloc'].iloc[-1]
        elif last_sig == 1 and last_info['pre_label'] == -1 and signal == 0:
            if df[col].iloc[-1] > df[col].iloc[last_sig_mm_loc]:
                signal = -1
                sig_type = '_正常Min'
                last_sig_mm_loc = \
                    df[df['mm'] == last_info['pre_label']]['iloc'].iloc[-1]

        # 信息保留至下一期
        if signal != 0:
            last_sig = signal
            last_sig_type = sig_type

        # 动态标签更新
        df.loc[it_idx, 'signal'] = signal
        data.loc[it_idx, 'signal'] = signal
        df.loc[it_idx, 'sig_type'] = sig_type
        data.loc[it_idx, 'sig_type'] = sig_type

        # 绘图
        plot_fig = True if plot_all else (True if signal != 0 else False)
        if fig_save_dir != False:
            if plot_fig:
                if fig_save_dir is None:
                    fig_save_path = None
                else:
                    fig_save_path = fig_save_dir + it_idx + '.png'
                if plot_lag == 'all':
                    df_plt = df
                else:
                    plot_lag = his_lag if plot_lag is None else plot_lag
                    df_plt = df.iloc[-plot_lag:, :]
                kwargs_plot_ = kwargs_plot.copy()
                idxs_mm1 = list(df_plt[df_plt['mm'] == 1].index)
                idxs_mm_1 = list(df_plt[df_plt['mm'] == -1].index)
                yparls_info_up = [(d, 'g', '-', 1.0) for d in idxs_mm1] + \
                                 [(d, 'r', '-', 1.0) for d in idxs_mm_1]
                title = it_idx + sig_type
                if 'title' in kwargs_plot_.keys():
                    title = kwargs_plot_['title'] + '_' + title
                    del kwargs_plot_['title']
                if 'cols_styl_up_left' in kwargs_plot_.keys():
                    cols_styl_up_left = kwargs_plot_['cols_styl_up_left']
                    del kwargs_plot_['cols_styl_up_left']
                else:
                    cols_styl_up_left = {}
                cols_styl_up_left.update({col: '.-b'})
                if 'cols_to_label_info' in kwargs_plot_.keys():
                    cols_to_label_info = kwargs_plot_['cols_to_label_info']
                    del kwargs_plot_['cols_to_label_info']
                else:
                    cols_to_label_info = {}
                cols_to_label_info.update(
                            {col: [['signal', (-1, 1), ('mo', 'co'), False]]})
                plot_series(df_plt, cols_styl_up_left=cols_styl_up_left,
                            cols_to_label_info=cols_to_label_info,
                            yparls_info_up=yparls_info_up,
                            title=title, fig_save_path=fig_save_path,
                            **kwargs_plot_,
                            )
                time.sleep(plot_sleep)

    return data['signal'], data['sig_type']

#%%
if __name__ == '__main__':
    from dramkit import TimeRecoder    
    from finfactory.load_his_data import load_index_joinquant

    tr = TimeRecoder()    
    
    #%%
    # '''
    arr = [1, 1, 1.3, 1.2, 2, 3, 4.8, 4.7, 5, 5]
    df1 = pd.DataFrame({'col': arr})
    df1['label'] = find_maxmin(df1['col'])
    plot_maxmins(df1, 'col', 'label',
                 title='标注极大极小值test1')
    OK, e = check_maxmins(df1, df1.columns[0], df1.columns[1])
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)

    df2 = pd.DataFrame({'col': arr[::-1]})
    df2['label'] = find_maxmin(df2['col'])
    plot_maxmins(df2, 'col', 'label',
                 title='标注极大极小值test2')
    OK, e = check_maxmins(df2, df2.columns[0], df2.columns[1])
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)
    # '''

    #%%
    # '''
    # 二次曲线叠加正弦余弦-------------------------------------------------------
    N = 200
    t = np.linspace(0, 1, N)
    s = 6*t*t + np.cos(10*2*np.pi*t*t) + np.sin(6*2*np.pi*t)
    df = pd.DataFrame(s, columns=['test'])

    # t_min = None
    t_min = 5
    df['label'] = find_maxmin(df['test'], t_min=t_min, min_dif_val=4, t_max=10)

    plot_maxmins(df, 'test', 'label',
                 title='标注极大极小值test：t_min='+str(t_min))

    OK, e = check_maxmins(df, df.columns[0], df.columns[1])
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)
    # '''
        
    #%%
    # '''
    # 趋势线叠加正弦余弦---------------------------------------------------------
    N = 200
    t = np.linspace(0, 1, N)
    trend = 6 * t
    circle1 = np.cos(10*2*np.pi*t*t)
    circle2 = np.sin(6*2*np.pi*t)
    # circle2 = 0 * t
    noise = np.random.randn(len(t)) / 5
    s = trend + circle1 + circle2 + noise
    df = pd.DataFrame({'trend': trend,
                       'circle1': circle1,
                       'circle2': circle2,
                       'noise': noise,
                       'series': s})

    # t_min = None
    t_min = 3
    df['label'] = find_maxmin(df['series'],
                              t_min=t_min,
                              min_dif_val=1,
                              t_max=10)

    plot_maxmins(df, 'series', 'label',
                 title='标注极大极小值test：t_min='+str(t_min))

    OK, e = check_maxmins(df, 'series', 'label')
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)
        
    
    plot_series(df,
                {'series': ('.-k', None),
                 'trend': ('-r', None),
                 'circle1': ('-b', None),
                 'circle2': ('-g', None),
                 'noise': ('-y', None)},
                # cols_to_label_info={'series':
                #     [['label', (1, -1), ('gv', 'r^'), False]]},
                title='周期波动示例')
    # '''

    #%%
    # '''
    # 50ETF日线行情------------------------------------------------------------
    fpath = '../_test/510500.SH_daily_qfq.csv'
    his_data = load_csv(fpath)
    his_data.set_index('date', drop=False, inplace=True)

    # N = his_data.shape[0]
    N = 200
    col = 'close'
    df = his_data.iloc[-N:, :].copy()

    # t_min = None
    t_min = 3
    df['label'] = find_maxmin(df[col], t_min=t_min)

    plot_maxmins(df.iloc[:, :], col, 'label', figsize=(12, 7))

    OK, e = check_maxmins(df, col, 'label')
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)
    # '''

    #%%
    '''
    # 上证50分钟行情------------------------------------------------------------
    his_data = load_index_joinquant('000016', '1min')
    his_data['time'] = his_data['time'].apply(lambda x:
                    x.replace('-', '').replace(':', '').replace(' ', '')[:-2])
    his_data.set_index('time', drop=False, inplace=True)
    his_data['ma'] = his_data['close'].rolling(20).mean()

    df = his_data.iloc[-240*18:-240*17, :].copy()
    N = df.shape[0]
    # N = 1000
    col = 'ma'
    df = df.iloc[-N:, :]

    # t_min = None
    series_ = df[col]
    t_min = 5
    df['label'] = find_maxmin(series_, t_min=t_min, min_dif_pct=0.15/100, t_max=10)

    plot_series(df.iloc[:-1, :], {'close': '.-k', col: '.-b'},
                cols_to_label_info={col:
                            [['label', (-1, 1), ('r^', 'gv'), False]]},
                n_xticks=6)

    OK, e = check_maxmins(df, col, 'label')
    if OK:
        print('极值点排列正确！')
    else:
        print('极值点排列错误:', e)

    his_cycle_info, df_his = get_his_maxmin_info(df, col, 'label')
    last_pos_info = get_last_pos_info(df.iloc[:-1, :], col, 'label', min_gap=0)
    # '''

    #%%
    # 动态标签
    # df['signal'], df['sig_type'] = find_maxmin_dy(df, col, t_min,
    #                                 his_lag=50,
    #                                 use_all=False,
    #                                 logger=None,
    #                                 n_log=10,
    #                                 # fig_save_dir='./_test/find_maxmin_dy/50ETF',
    #                                 fig_save_dir=None,
    #                                 plot_all=True,
    #                                 plot_lag='all',
    #                                 kwargs_findMaxMin={},
    #                                 kwargs_plot={
    #                                     'cols_styl_up_left': {'close': '.-k'},
    #                                     'cols_to_label_info': {'close':
    #                                         [['signal', (-1, 1), ('r^', 'gv'),
    #                                           False]]},
    #                                     'title': '50ETF',
    #                                     'n_xticks': 4,
    #                                     'figsize': (10, 5)},
    #                                 plot_sleep=0.5)

    #%%
    tr.used()
