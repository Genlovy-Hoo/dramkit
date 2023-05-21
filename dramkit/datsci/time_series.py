# -*- coding: utf-8 -*-

'''
时间序列处理工具函数

TODO
----
改成class以简化函数调用传参
'''

import numpy as np
import pandas as pd
from dramkit.gentools import con_count, isnull
from dramkit.logtools.utils_logger import logger_show

#%%
def fillna_ma(series, ma=None, ma_min=1):
    '''
    | 用移动平均ma填充序列series中的缺失值
    | ma设置填充时向前取平均数用的期数，ma_min设置最小期数
    | 若ma为None，则根据最大连续缺失记录数确定ma期数
    '''

    if series.name is None:
        series.name = 'series'
    col = series.name
    df = pd.DataFrame(series)

    if isnull(ma):
        tmp = con_count(series, lambda x: True if isnull(x) else False)
        ma = 2 * tmp.max()
        ma = max(ma, ma_min*2)

    df[col+'_ma'] = df[col].rolling(ma, ma_min).mean()

    df[col] = df[[col, col+'_ma']].apply(lambda x:
               x[col] if not isnull(x[col]) else \
               (x[col+'_ma'] if not isnull(x[col+'_ma']) else x[col]), axis=1)

    return df[col]

#%%
def get_directional_accuracy_1step(y_true, y_pred):
    '''
    | 一步向前预测方向准确率计算
    | y_true和y_pred为pd.Series
    '''

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['y_true_last'] = df['y_true'].shift(1)

    df['d_true'] = df['y_true'] - df['y_true_last']
    df['d_true'] = df['d_true'].apply(lambda x: 1 if x > 0 \
                                      else (-1 if x < 0 else 0))
    df['d_pred'] = df['y_pred'] - df['y_true_last']
    df['d_pred'] = df['d_pred'].apply(lambda x: 1 if x > 0 \
                                      else (-1 if x < 0 else 0))

    return (df['d_true'] == df['d_pred']).sum() / df.shape[0]

#%%
# X二维，ycol一维一步向前预测情况
def genXy_X2d_y1d1step(df, ycol, Xcols_series_lag=None, Xcols_other=None,
                       gap=0):
    '''
    构造时间序列预测的输入X和输出y，适用于X二维，ycol一维一步向前预测情况
    
    TODO
    ----
    gap为-1的情况检查确认

    Parameters
    ----------
    df : pd.DataFrame
        包含因变量列和所有自变量列的数据集
    ycol : str
        因变量序列名
    Xcols_series_lag : list, None
        | 序列自变量列名和历史期数，格式如：
        | [(xcol1, lag1), (xcol2, lag2), ...]
        | 若为None，则默认设置为[(ycol, 3)]
    Xcols_other : list, None
        非序列自变量列名列表

        .. note:: 
            序列自变量视为不能拿到跟因变量预测同期的数据，非序列自变量视为可以拿到同期数据
    gap : int
        预测跳跃期数（eg.用截止今天的数据预测gap天之后的数据）

    Returns
    -------
    X : np.array
        二维np.array，行数为样本量，列数为特征数量
    y : np.array
        一维np.array，长度等于样本量
    '''

    y_series = np.array(df[ycol])
    if Xcols_series_lag is None:
        Xcols_series_lag = [(ycol, 3)]
    if Xcols_other is not None:
        X_other = np.array(df[Xcols_other])

    X, y =[], []

    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]
    for k in range(len(y_series)-y_lag-gap):
        X_part1 = []
        for Xcol_series, lag in Xcols_series_lag:
            x_tmp = np.array(df[Xcol_series])[k+y_lag-lag: k+y_lag]
            X_part1.append(x_tmp.reshape(1, -1))
        X_part1 = np.concatenate(X_part1, axis=1)

        if Xcols_other is not None:
            X_part2 = X_other[k+y_lag+gap].reshape(1, -1)
            X.append(np.concatenate((X_part1, X_part2), axis=1))
        else:
            X.append(X_part1)

        y.append(y_series[k+y_lag+gap])

    X, y = np.concatenate(X, axis=0), np.array(y)

    return X, y


def predict_X2d_y1d1step(model, func_pred, df_pre, ycol, Xcols_series_lag,
                         Xcols_other, gap, **kwargs):
    '''
    | 使用训练好的模型在数据df_pre上进行构造样本并预测(非滚动预测)
    | 适用于X二维、ycol一维一步向前预测情况

    Parameters
    ----------
    model :
        训练好的预测模型
    func_pred : function
        func_pred为预测函数，接受必要参数model和X，也接受可选参数**kwargs
    df_pre : pd.DataFrame
        待预测所用数据，其中 ``ycol`` 为待预测列
    Xcols_series_lag, Xcols_other, gap : 
        参数意义同 :func:`genXy_X2d_y1d1step` 中参数意义


    :returns: `pd.DataFrame` - 返回pd.DataFrame包含原来的数据以及预测结果ycol+'_pre'列
    '''

    df_pre = df_pre.copy()
    df_pre[ycol+'_pre'] = np.nan

    X, _ = genXy_X2d_y1d1step(df_pre, ycol, Xcols_series_lag=Xcols_series_lag,
                              Xcols_other=Xcols_other, gap=gap)
    y_pre = func_pred(model, X, **kwargs)

    df_pre.loc[df_pre.index[-len(y_pre):], ycol+'_pre'] = y_pre

    return df_pre


def valid_test_predict_X2d_y1d1step(model, func_pred, df_train, df_valid,
                                    df_test, ycol, Xcols_series_lag,
                                    Xcols_other, gap, **kwargs):
    '''
    | 用训练好的模型做时间序列验证集和测试集预测(非滚动预测, 即不使用前面预测的数据作为输入)
    | 适用于X二维、ycol一维一步向前预测情况

    Parameters
    ----------
    model, func_pred, ycol, Xcols_series_lag, Xcols_other, gap, **kwargs : 
        与 :func:`predict_X2d_y1d1step` 参数意义相同
    df_train : pd.DataFrame
        df_train为用作训练的数据
    df_valid : pd.DataFrame
        df_valid为验证集待预测数据
    df_test : pd.DataFrame
        df_test为测试集待预测数据

        Note
        ----
        | df_valid和df_test的列须与df_train相同
        | df_valid和df_test可以为None或pd.DataFrame
        | df_train、df_valid、df_test须为连续时间序列
        | 当df_test不为空时，df_valid的ycol列必须有真实值

    Returns
    -------
    valid_pre : pd.DataFrame
        验证集预测结果，包含ycol+'_pre'列
    test_pre : pd.DataFrame
        测试集预测结果，包含ycol+'_pre'列
    '''

    def _is_effect(df):
        if df is None or df.shape[0] == 0:
            return False
        elif df.shape[0] > 0:
            return True

    if not _is_effect(df_valid) and not _is_effect(df_test):
        return None, None

    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]

    # 判断df_train和df_valid样本量是否满足要求
    if _is_effect(df_valid) and df_train.shape[0] < y_lag+gap:
        raise ValueError(
              '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(y_lag+gap))
    if not _is_effect(df_valid) and _is_effect(df_test) and \
                                            df_train.shape[0] < y_lag+gap:
        raise ValueError(
              '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(y_lag+gap))

    # 构造验证集预测用df
    if _is_effect(df_valid):
        valid_pre = pd.concat((df_train.iloc[-y_lag-gap:, :], df_valid),
                              axis=0).reindex(columns=df_valid.columns)
    else:
        valid_pre = None
    # 构造测试集预测用df
    if _is_effect(df_test) and _is_effect(df_valid):
        tmp = pd.concat((df_train, df_valid), axis=0)
        test_pre = pd.concat((tmp.iloc[-y_lag-gap:, :], df_test), axis=0)
        test_pre = test_pre.reindex(columns=df_test.columns)
    elif _is_effect(df_test) and not _is_effect(df_valid):
        test_pre = pd.concat((df_train.iloc[-y_lag-gap:, :], df_test), axis=0)
        test_pre = test_pre.reindex(columns=df_test.columns)
    else:
        test_pre = None

    # 验证集预测
    if valid_pre is not None:
        valid_pre = predict_X2d_y1d1step(model, func_pred, valid_pre, ycol,
                                Xcols_series_lag, Xcols_other, gap, **kwargs)
        valid_pre = valid_pre.iloc[y_lag+gap:y_lag+gap+df_valid.shape[0], :]
    # 测试集预测
    if test_pre is not None:
        test_pre = predict_X2d_y1d1step(model, func_pred, test_pre, ycol,
                                Xcols_series_lag, Xcols_other, gap, **kwargs)
        test_pre = test_pre.iloc[y_lag+gap:y_lag+gap+df_test.shape[0], :]

    return valid_pre, test_pre


def forward_predict_X2d_y1d1step(model, func_pred, df_pre, ycol,
                                 Xcols_series_lag, Xcols_other, gap,
                                 **kwargs):
    '''
    | 用训练好的模型做在数据df_pre上进行构造样本并进行向前滚动预测(前面的预测作为后面的输入)
    | 适用于X二维、ycol一维一步向前预测情况
    
    TODO
    ----
    gap为-1的情况检查确认

    Parameters
    ----------
    model : 
        训练好的预测模型
    func_pred : function
        func_pred为预测函数，接受必要参数model和X，也接受可选参数**kwargs
    df_pre : pd.DataFrame
        待预测所用数据，其中 ``ycol`` 为待预测列。
        除了初始构造输入X必须满足的数据行外，后面的数据可以为空值
    Xcols_series_lag, Xcols_other, gap : 
        参数意义同 :func:`genXy_X2d_y1d1step` 中参数意义，
        用其构造输入变量的过程亦完全相同
        

    :returns: `pd.DataFrame` - 返回pd.DataFrame包含原来的数据以及预测结果ycol+'_pre'列
    '''

    df_pre = df_pre.copy()

    if Xcols_other is not None:
        X_other = np.array(df_pre[Xcols_other])

    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]

    df_pre[ycol+'_pre'] = df_pre[ycol].copy()
    if gap >= 0:
        for k in range(y_lag+gap, df_pre.shape[0]):
            df_pre.loc[df_pre.index[k], ycol+'_pre'] = np.nan
    Xcols_series = [x[0] for x in Xcols_series_lag]
    if ycol in Xcols_series:
        ycol_idx = Xcols_series.index(ycol)
        ycol_lag = Xcols_series_lag[ycol_idx][1]
        Xcols_series_lag_new = Xcols_series_lag.copy()
        Xcols_series_lag_new[ycol_idx] = (ycol+'_pre', ycol_lag)
    else:
        Xcols_series_lag_new = Xcols_series_lag.copy()

    for k in range(0, df_pre.shape[0]-y_lag-gap):
        X_part1 = []
        for Xcol_series, lag in Xcols_series_lag_new:
            x_tmp = np.array(df_pre[Xcol_series])[k+y_lag-lag:k+y_lag]
            X_part1.append(x_tmp.reshape(1, -1))
        X_part1 = np.concatenate(X_part1, axis=1)

        if Xcols_other is not None:
            X_part2 = X_other[k+y_lag+gap].reshape(1, -1)
            X = np.concatenate((X_part1, X_part2), axis=1)
        else:
            X = X_part1

        pred = func_pred(model, X, **kwargs)[0]

        df_pre.loc[df_pre.index[k+y_lag+gap], ycol+'_pre'] = pred

    return df_pre


def forward_valid_test_predict_X2d_y1d1step(model, func_pred, df_train, df_valid,
                                            df_test, ycol, Xcols_series_lag,
                                            Xcols_other, gap, **kwargs):
    '''
    | 用训练好的模型做时间序列验证集和测试集预测(滚动预测, 前面的预测作为后面的输入)
    | 适用于X二维、ycol一维一步向前预测情况

    Parameters
    ----------
    model, func_pred, ycol, Xcols_series_lag, Xcols_other, gap, **kwargs :
        与 :func:`forward_predict_X2d_y1d1step` 参数意义相同
    df_train : pd.DataFrame
        df_train为用作训练的数据
    df_valid : pd.DataFrame
        df_valid为验证集待预测数据
    df_test : pd.DataFrame
        df_test为测试集待预测数据

        Note
        ----
        | df_valid和df_test的列须与df_train相同
        | df_valid和df_test可以为None或pd.DataFrame
        | df_train、df_valid、df_test须为连续时间序列
        | 当df_test不为空时，df_valid的ycol列必须有真实值

    Returns
    -------
    valid_pre : pd.DataFrame
        验证集预测结果，包含ycol+'_pre'列
    test_pre : pd.DataFrame
        测试集预测结果，包含ycol+'_pre'列
    '''

    def _is_effect(df):
        if df is None or df.shape[0] == 0:
            return False
        elif df.shape[0] > 0:
            return True

    if not _is_effect(df_valid) and not _is_effect(df_test):
        return None, None

    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]

    # 判断df_train和df_valid样本量是否满足要求
    if _is_effect(df_valid) and df_train.shape[0] < y_lag+gap:
        raise ValueError(
              '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(y_lag+gap))
    if not _is_effect(df_valid) and _is_effect(df_test) and \
                                            df_train.shape[0] < y_lag+gap:
        raise ValueError(
              '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(y_lag+gap))

    # 构造验证集预测用df
    if _is_effect(df_valid):
        valid_pre = pd.concat((df_train.iloc[-y_lag-gap:, :], df_valid),
                              axis=0).reindex(columns=df_valid.columns)
    else:
        valid_pre = None
    # 构造测试集预测用df
    if _is_effect(df_test) and _is_effect(df_valid):
        tmp = pd.concat((df_train, df_valid), axis=0)
        test_pre = pd.concat((tmp.iloc[-y_lag-gap:, :], df_test), axis=0)
        test_pre = test_pre.reindex(columns=df_test.columns)
    elif _is_effect(df_test) and not _is_effect(df_valid):
        test_pre = pd.concat((df_train.iloc[-y_lag-gap:, :], df_test), axis=0)
        test_pre = test_pre.reindex(columns=df_test.columns)
    else:
        test_pre = None

    # 验证集预测
    if valid_pre is not None:
        valid_pre = forward_predict_X2d_y1d1step(model, func_pred, valid_pre,
                            ycol, Xcols_series_lag, Xcols_other, gap, **kwargs)
        valid_pre = valid_pre.iloc[y_lag+gap:y_lag+gap+df_valid.shape[0], :]
    # 测试集预测
    if test_pre is not None:
        test_pre = forward_predict_X2d_y1d1step(model, func_pred, test_pre,
                            ycol, Xcols_series_lag, Xcols_other, gap, **kwargs)
        test_pre = test_pre.iloc[y_lag+gap:y_lag+gap+df_test.shape[0], :]

    return valid_pre, test_pre

#%%
# X二维，ycol一维多步向前预测情况
def genXy_X2d_y1dsteps(df, ycol, Xcols_series_lag=None, Xcols_other=None,
                       y_step=1, gap=0):
    '''    
    构造时间序列预测的输入X和输出y，适用于X二维，ycol一维多步向前预测情况
    
    TODO
    ----
    gap为-1的情况检查确认

    Parameters
    ----------
    df : pd.DataFrame
        包含因变量列和所有自变量列的数据集
    ycol : str
        因变量序列名
    Xcols_series_lag : list, None
        | 序列自变量列名和历史期数，格式如：
        | [(xcol1, lag1), (xcol2, lag2), ...]
        | 若为None，则默认设置为[(ycol, 3)]
    Xcols_other : list, None
        非序列自变量列名列表

        .. note:: 
            序列自变量视为不能拿到跟因变量预测同期的数据，非序列自变量视为可以拿到同期数据
    y_step : int
        一次向前预测ycol的步数
        （即因变量y的维度，多输出模型中y为多维，时间序列上相当于向前预测多步）
    gap : int
        预测跳跃期数（eg.用截止今天的数据预测gap天之后的y_step天的数据）
        
    Returns
    -------
    X : np.array
        二维，行数为样本量，列数为特征数量
    y : np.array
        若y_step为1则为一维，若y_step大于1，则形状为样本量*y_step
    '''

    y_series = np.array(df[ycol])
    if Xcols_series_lag is None:
        Xcols_series_lag = [(ycol, 3)]
    if Xcols_other is not None:
        X_other = np.array(df[Xcols_other])

    X, y = [], []

    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]
    for k in range(len(y_series)-y_lag-gap-y_step+1):
        X_part1 = []
        for Xcol_series, lag in Xcols_series_lag:
            x_tmp = np.array(df[Xcol_series])[k+y_lag-lag: k+y_lag]
            X_part1.append(x_tmp.reshape(1, -1))
        X_part1 = np.concatenate(X_part1, axis=1)

        if Xcols_other is not None:
            X_part2 = X_other[k+y_lag+gap: k+y_lag+gap+y_step]
            X_part2 = X_part2.transpose().reshape(1, -1)
            X.append(np.concatenate((X_part1, X_part2), axis=1))
        else:
            X.append(X_part1)

        y.append(y_series[k+y_lag+gap: k+y_lag+gap+y_step])

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    if y_step == 1:
        y = y.reshape(-1)

    return X, np.array(y)


def forward_predict_X2d_y1dsteps(model, func_pred, df_pre, ycol,
                                 Xcols_series_lag, Xcols_other, y_step, gap,
                                 **kwargs):
    '''
    | 用训练好的模型做在数据df_pre上进行构造样本并进行向前滚动预测(前面的预测作为后面的输入)
    | 适用于X二维，ycol一维多步向前预测情况
    
    TODO
    ----
    gap为-1的情况检查确认
    
    Parameters
    ----------
    model : 
        训练好的预测模型
    func_pred : function
        func_pred为预测函数，接受必要参数model和X，也接受可选参数**kwargs
    df_pre : pd.DataFrame
        待预测所用数据，其中 ``ycol`` 为待预测列。
        除了初始构造输入X必须满足的数据行外，后面的数据可以为空值
    Xcols_series_lag, Xcols_other, y_step, gap : 
        参数意义同 :func:`genXy_X2d_y1dsteps` 中参数意义，
        用其构造输入变量的过程亦完全相同

    Returns
    -------
    df_pre : pd.DataFrame
        返回数据中包含原来的数据以及预测结果ycol+'_pre'列
    '''

    df_pre = df_pre.copy()

    if Xcols_other is not None:
        X_other = np.array(df_pre[Xcols_other])
    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]

    if (df_pre.shape[0]-y_lag-gap) % y_step != 0:
        raise ValueError('必须保证`(df_pre.shape[0]-y_lag) % y_step == 0`！')

    df_pre[ycol+'_pre'] = df_pre[ycol].copy()
    for k in range(y_lag+gap, df_pre.shape[0]):
        df_pre.loc[df_pre.index[k], ycol+'_pre'] = np.nan
    Xcols_series = [x[0] for x in Xcols_series_lag]
    if ycol in Xcols_series:
        ycol_idx = Xcols_series.index(ycol)
        ycol_lag = Xcols_series_lag[ycol_idx][1]
        Xcols_series_lag_new = Xcols_series_lag.copy()
        Xcols_series_lag_new[ycol_idx] = (ycol+'_pre', ycol_lag)

    for k in range(0, df_pre.shape[0]-y_lag, y_step):
        X_part1 = []
        for Xcol_series, lag in Xcols_series_lag_new:
            x_tmp = np.array(df_pre[Xcol_series])[k+y_lag-lag:k+y_lag]
            X_part1.append(x_tmp.reshape(1, -1))
        X_part1 = np.concatenate(X_part1, axis=1)

        if Xcols_other is not None:
            X_part2 = X_other[k+y_lag+gap: k+y_lag+gap+y_step]
            X_part2 = X_part2.transpose().reshape(1, -1)
            X = np.concatenate((X_part1, X_part2), axis=1)
        else:
            X = X_part1

        pred = func_pred(model, X, **kwargs)[0]

        df_pre.loc[df_pre.index[k+y_lag+gap: k+y_lag+gap+y_step],
                                                       ycol+'_pre'] = pred

    return df_pre


def forward_valid_test_predict_X2d_y1dsteps(model, func_pred, df_train,
                                            df_valid, df_test, ycol,
                                            Xcols_series_lag, Xcols_other,
                                            y_step, gap, logger=None,
                                            **kwargs):
    '''
    | 用训练好的模型做时间序列验证集和测试集预测(滚动预测, 前面的预测作为后面的输入)
    | 适用于X二维，ycol一维多步向前预测情况

    Parameters
    ----------
    model, func_pred, ycol, Xcols_series_lag, Xcols_other, y_step, gap, **kwargs : 
        与 :func:`forward_predict_X2d_y1dsteps` 中参数意义相同
    df_train : pd.DataFrame
        df_train为用作训练的数据
    df_valid : pd.DataFrame
        df_valid为验证集待预测数据
    df_test : pd.DataFrame
        df_test为测试集待预测数据

        Note
        ----
        | df_valid和df_test的列须与df_train相同
        | df_valid和df_test可以为None或pd.DataFrame
        | df_train、df_valid、df_test须为连续时间序列
        | 当df_test不为空时，df_valid的ycol列必须有真实值

    Returns
    -------
    valid_pre : pd.DataFrame
        验证集预测结果，包含ycol+'_pre'列
    test_pre : pd.DataFrame
        测试集预测结果，包含ycol+'_pre'列
    '''

    def _is_effect(df):
        if df is None or df.shape[0] == 0:
            return False
        elif df.shape[0] > 0:
            return True

    if not _is_effect(df_valid) and not _is_effect(df_test):
        return None, None

    y_lag = max(Xcols_series_lag, key=lambda x: x[1])[1]

    # 判断df_train和df_valid样本量是否满足要求
    if _is_effect(df_valid) and df_train.shape[0] < y_lag+gap:
        raise ValueError(
            '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(y_lag+gap))
    if not _is_effect(df_valid) and _is_effect(df_test) and \
                                            df_train.shape[0] < y_lag+gap:
        raise ValueError(
            '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(y_lag+gap))

    # 构造验证集预测用df
    if _is_effect(df_valid):
        n_valid = df_valid.shape[0]
        if n_valid % y_step == 0:
            valid_pre = pd.concat((df_train.iloc[-y_lag-gap:, :], df_valid),
                                  axis=0).reindex(columns=df_valid.columns)
        else:
            # 处理预测步长与样本量冲突问题
            n_ext = y_step - n_valid % y_step
            if df_train.shape[0] >= y_lag+gap+y_step:
                logger_show('用训练集数据扩充验证集使得其样本数能被y_step整除！',
                            logger, 'warning')
                df_valid_tmp = pd.concat((df_valid, df_train.iloc[:n_ext, :]),
                                         axis=0)
                valid_pre = pd.concat(
                    (df_train.iloc[-y_lag-gap:, :], df_valid_tmp), axis=0)
                valid_pre = valid_pre.reindex(columns=df_valid.columns)
            else:
                raise ValueError(
                 '向前多步预测时验证集样本量不能被y_step整除且无法用训练集数据扩充！')
    else:
        valid_pre = None
    # 构造测试集预测用df
    if _is_effect(df_test) and _is_effect(df_valid):
        n_test = df_test.shape[0]
        tmp = pd.concat((df_train, df_valid), axis=0)
        if n_test % y_step == 0:
            test_pre = pd.concat((tmp.iloc[-y_lag-gap:, :], df_test), axis=0)
            test_pre = test_pre.reindex(columns=df_test.columns)
        else:
            # 处理预测步长与样本量冲突问题
            n_ext = y_step - n_test % y_step
            if tmp.shape[0] >= y_lag+gap+y_step:
                logger_show(
                    '用训练集和验证集数据扩充测试集使得其样本数能被y_step整除！',
                    logger, 'warning')
                df_test_tmp = pd.concat((df_test, tmp.iloc[:n_ext, :]), axis=0)
                test_pre = pd.concat(
                    (tmp.iloc[-y_lag-gap:, :], df_test_tmp), axis=0)
                test_pre = test_pre.reindex(columns=df_test.columns)
            else:
                raise ValueError(
           '向前多步预测时训练集样本量不能被y_step整除且无法用训练集和验证集数据扩充！')
    elif _is_effect(df_test) and not _is_effect(df_valid):
        n_test = df_test.shape[0]
        if n_test % y_step == 0:
            test_pre = pd.concat((df_train.iloc[-y_lag-gap:, :], df_test),
                                 axis=0).reindex(columns=df_test.columns)
        else:
            # 处理预测步长与样本量冲突问题
            n_ext = y_step - n_test % y_step
            if df_train.shape[0] >= y_lag+gap+y_step:
                logger_show('用训练集数据扩充测试集使得其样本数能被y_step整除！',
                            logger, 'warning')
                df_test_tmp = pd.concat((df_test, df_train.iloc[:n_ext, :]),
                                        axis=0)
                test_pre = pd.concat(
                    (df_train.iloc[-y_lag-gap:, :], df_test_tmp), axis=0)
                test_pre = test_pre.reindex(columns=df_test.columns)
            else:
                raise ValueError(
                 '向前多步预测时验证集样本量不能被y_step整除且无法用训练集数据扩充！')
    else:
        test_pre = None

    # 验证集预测
    if valid_pre is not None:
        valid_pre = forward_predict_X2d_y1dsteps(model, func_pred, valid_pre,
                    ycol, Xcols_series_lag, Xcols_other, y_step, gap, **kwargs)
        valid_pre = valid_pre.iloc[y_lag+gap:y_lag+gap+n_valid, :]
    # 测试集预测
    if test_pre is not None:
        test_pre = forward_predict_X2d_y1dsteps(model, func_pred, test_pre,
                   ycol, Xcols_series_lag, Xcols_other, y_step, gap, **kwargs)
        test_pre = test_pre.iloc[y_lag+gap:y_lag+gap+n_test, :]

    return valid_pre, test_pre

#%%
if __name__ == '__main__':
    from dramkit import plot_series
    from sklearn.linear_model import LinearRegression as lr
    from sklearn.svm import SVR as svr
    from dramkit.gentools import cut_range_to_subs, replace_repeat_pd, TimeRecoder
    from dramkit._tmp.utils_SignalDec import dec_emds, merge_high_modes
    from dramkit._tmp.utils_SignalDec import plot_modes
    from finfactory.fintools import get_yield_curve
    from finfactory.load_his_data import load_future_daily_tushare
    from pprint import pprint


    tr = TimeRecoder()

    #%%
    code = 'IF.CFX'
    data = load_future_daily_tushare(code)

    data.set_index('date', drop=False, inplace=True)

    N = data.shape[0]
    data = data.iloc[-N:, :].copy()

    plot_series(data, {'close': ('.-k', 'close')},
                figsize=(9.5, 7), fontname='Times New Roman', grids=True)


    dates = list(data['date'].unique())
    dates.sort()

    modes_all = dec_emds(data['close'])
    modes_all = merge_high_modes(modes_all)
    plot_modes(modes_all)

    dates_test = [x for x in dates if x >= '2020-01-01']

    def func_mdl_pre(model, X):
        return model.predict(X)

    #%%
    # ycol = 'close'
    # lag = 20
    # Xcols_series_lag = [('close', lag), ('high', lag), ('low', lag),
    #                     ('open', lag)]
    # Xcols_other = None
    # gap = 0


    # # 滚动训练-预测
    # df_pre = []
    # k = 0
    # idxs = cut_range_to_subs(len(dates_test), gap=100)
    # for idx1, idx2 in idxs[:]:
    #     k += 1
    #     if k % 10 == 0:
    #         print(k, '/', len(idxs))
    #     df_train = data[data['date'] < dates_test[idx1]]
    #     df_valid = data[(data['date'] >= dates_test[idx1]) & \
    #                     (data['date'] <= dates_test[idx2-1])]

    #     X, y = genXy_X2d_y1d1step(df_train, ycol,
    #                               Xcols_series_lag=Xcols_series_lag,
    #                               Xcols_other=Xcols_other,
    #                               gap=gap)
    #     mdl = svr().fit(X, y)
    #     valid_pre, _ = valid_test_predict_X2d_y1d1step(mdl, func_mdl_pre,
    #                                     df_train, df_valid, None, ycol,
    #                                     Xcols_series_lag, Xcols_other, gap)
    #     df_pre.append(valid_pre)
    # df_pre = pd.concat(df_pre, axis=0)

    # plot_series(df_pre, {'close': ('.-b', None), 'close_pre': ('.-r', None)},
    #             figsize=(9.5, 7), fontname='Times New Roman', grids=True)

    # da = get_directional_accuracy_1step(df_pre[ycol], df_pre[ycol+'_pre'])
    # print('方向准确率: {}'.format(da))

    #%%
    df_pre = []
    k = 0
    idxs = cut_range_to_subs(len(dates_test), gap=1)
    for idx1, idx2 in idxs[:]:
        k += 1
        if k % 10 == 0:
            print(k, '/', len(idxs))
        df_train = data[data['date'] < dates_test[idx1]].copy()
        df_valid = data[(data['date'] >= dates_test[idx1]) & \
                        (data['date'] <= dates_test[idx2-1])].copy()

        modes_pre = df_valid.copy()
        modes_pre['close_pre'] = 0

        modes = dec_emds(df_train['close'])
        modes = merge_high_modes(modes)
        # plot_modes(modes)

        df_train = pd.merge(df_train, modes, how='outer',
                            left_index=True, right_index=True)

        col_modes = modes.columns
        for col in col_modes:
            df_valid[col] = np.nan

        for col in col_modes:
            # if 'IMF' not in col:
            ycol = col
            lag, gap = 10, 0
            Xcols_series_lag = [(col, lag)]
            Xcols_other = None
            X, y = genXy_X2d_y1d1step(df_train, ycol,
                                      Xcols_series_lag=Xcols_series_lag,
                                      Xcols_other=Xcols_other,
                                      gap=gap)
            mdl = lr().fit(X, y)
            valid_pre, _ = valid_test_predict_X2d_y1d1step(mdl, func_mdl_pre,
                                        df_train, df_valid, None, ycol,
                                        Xcols_series_lag, Xcols_other, gap)

            modes_pre['close_pre'] = modes_pre['close_pre'] + \
                                      valid_pre[ycol+'_pre']
        df_pre.append(modes_pre)
    df_pre = pd.concat(df_pre, axis=0)

    df_pre.to_csv('./_test/df_pre.csv')

    plot_series(df_pre, {'close': ('.-b', None), 'close_pre': ('.-r', None)},
                figsize=(9.5, 7), fontname='Times New Roman', grids=True)

    da = get_directional_accuracy_1step(df_pre['close'], df_pre['close_pre'])
    print('方向准确率: {}'.format(da))

    #%%
    df_pre['close_last'] = df_pre['close'].shift(1)
    df_pre['signal_open'] = df_pre[['close_pre', 'close_last']].apply(
                lambda x: -1 if x['close_pre']-x['close_last'] > 0 else \
                    (1 if x['close_pre']-x['close_last'] < 0 else 0), axis=1)
    df_pre['signal_close'] = -df_pre['signal_open']

    tmp1 = df_pre.reindex(columns=['date', 'open', 'signal_open'])
    tmp1['time'] = tmp1['date'] + ' 09:25:00'
    tmp1.set_index('time', drop=True, inplace=True)
    tmp1.drop('date', axis=1, inplace=True)
    tmp2 = df_pre.reindex(columns=['date', 'close', 'signal_close'])
    tmp2['time'] = tmp2['date'] + ' 15:00:00'
    tmp2.set_index('time', drop=True, inplace=True)
    tmp2.drop('date', axis=1, inplace=True)

    df1 = pd.merge(tmp1, tmp2, how='outer', left_index=True, right_index=True)
    df1.sort_index(ascending=True, inplace=True)
    df1['signal'] = df1['signal_open'].fillna(0) + \
                                                df1['signal_close'].fillna(0)
    df1['price'] = df1['open'].fillna(0) + df1['close'].fillna(0)
    df1 = df1.reindex(columns=['price', 'signal'])

    trade_gain_info1, df_gain1 = get_yield_curve(df1, 'signal',
                                            col_price='price',
                                            col_price_buy='price',
                                            col_price_sel='price',
                                            nn=500,
                                            base_money=None, base_vol=1,
                                            fee=0.2/1000, max_loss=None,
                                            max_gain=None, max_down=None,
                                            func_vol_stoploss=lambda x, y, a, b, c: 0,
                                            init_cash=0,
                                            force_final0='settle',
                                            kwargs_plot={'title': code,
                                                         'n_xticks': 6})

    pprint(trade_gain_info1)

    #%%
    df_pre['close_last'] = df_pre['close'].shift(1)
    df_pre['signal'] = df_pre[['close_pre', 'close_last']].apply(
                lambda x: -1 if x['close_pre']-x['close_last'] > 0 else \
                    (1 if x['close_pre']-x['close_last'] < 0 else 0), axis=1)
    df_pre['signal'] = replace_repeat_pd(df_pre['signal'], 1, 0)
    df_pre['signal'] = replace_repeat_pd(df_pre['signal'], -1, 0)
    df2 = df_pre.reindex(columns=['open', 'close', 'signal'])

    trade_gain_info2, df_gain2 = get_yield_curve(df2, 'signal',
                                            col_price='close',
                                            col_price_buy='open',
                                            col_price_sel='open',
                                            nn=250,
                                            func_vol_sub='hold_base_1',
                                            base_money=None, base_vol=1,
                                            fee=0.2/1000, max_loss=None,
                                            max_gain=None, max_down=None,
                                            func_vol_stoploss=lambda x, y, a, b, c: 0,
                                            init_cash=0,
                                            force_final0='settle',
                                            kwargs_plot={'title': code,
                                                         'n_xticks': 6})

    pprint(trade_gain_info2)

    #%%
    df_pre['close_last'] = df_pre['close'].shift(1)
    df_pre['signal'] = df_pre[['close_pre', 'close_last']].apply(
                lambda x: -1 if x['close_pre']-x['close_last'] > 0 else \
                    (1 if x['close_pre']-x['close_last'] < 0 else 0), axis=1)
    df_pre['signal'] = df_pre['signal'].shift(-1)
    df_pre['signal'] = df_pre['signal'].fillna(0)
    df_pre['signal'] = replace_repeat_pd(df_pre['signal'], 1, 0)
    df_pre['signal'] = replace_repeat_pd(df_pre['signal'], -1, 0)
    df3 = df_pre.reindex(columns=['close', 'signal'])

    trade_gain_info3, df_gain3 = get_yield_curve(df3, 'signal',
                                            col_price='close',
                                            col_price_buy='close',
                                            col_price_sel='close',
                                            nn=250,
                                            func_vol_sub='hold_base_1',
                                            base_money=None, base_vol=1,
                                            fee=0.2/1000, max_loss=1/100,
                                            max_gain=None, max_down=None,
                                            func_vol_stoploss=lambda x, y, a, b, c: 0,
                                            init_cash=0,
                                            force_final0='settle',
                                            kwargs_plot={'title': code,
                                                         'n_xticks': 6})

    pprint(trade_gain_info3)

    #%%
    tr.used()
