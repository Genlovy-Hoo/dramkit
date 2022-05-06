# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from dramkit.logtools.utils_logger import logger_show

#%%
# X二维，ycols多维多步向前预测情况
def genXy_multi_steps(df, ycols, Xcols_series_lag=None, Xcols_other=None,
                      gap=0, y_step=1):
    '''
    构造时间序列预测的输入X和输出y，X二维，ycols多维向前多步预测情况
    ycols为因变量序列名列表，Xcols_series_lag为序列自变量列名和期数，格式如：
        [(xcol1, lag1), (xcol2, lag2)]，若为None，则默认设置为[(ycols, 3)]
    Xcols_other为非序列自变量列名列表
    gap为预测滞后期数（即用截止今天的数据预测gap天之后的y_step天的数据）
    y_step为因变量y维度，默认为1（多输出模型中y为多维，时间序列上相当于向前预测多步）
    注：序列自变量视为不能拿到跟因变量预测同期的数据，非序列自变量视为可以拿到同期数据
    '''

    y_series = np.array(df[ycols])
    if Xcols_series_lag is None:
        Xcols_series_lag = [(ycols, 3)]
    if Xcols_other is not None:
        X_other = np.array(df[Xcols_other])

    X, y =[], []

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


        y.append(y_series[k+y_lag+gap: k+y_lag+gap+y_step].reshape(1, -1))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    if y_step == 1 and len(ycols) == 1:
        y = y.reshape(-1)

    return X, np.array(y)

#%%
# X三维情况
def SeriesPredict_X3d_steps(model, funcPred, df_train, df_valid, df_test,
                            ycol, Xcols, lag, gap, y_step, logger=None,
                            **kwargs):
    '''
    X三维，ycol一维多步向前预测情况
    时间序列验证集和测试集向前滚动预测（前面的预测作为后面的输入）
    df_train为用作训练的数据，df_valid和df_test为验证集和测试集数据，
    df_valid和df_test的列须与df_train相同
    df_valid和df_test可以为None或pd.DataFrame
    df_train、df_valid、df_test须为连续时间序列，当df_test不为空时，df_valid的ycol
    列必须有真实值
    ycol, Xcols, lag, gap, y_step, **kwargs参数与
    forward_predict_X3d_steps中参数意义相同
    y_step为一次向前预测的步数（多输出模型对应到时间序列上可一次做多步预测）
    '''

    def isEffect(df):
        if df is None or df.shape[0] == 0:
            return False
        elif df.shape[0] > 0:
            return True

    if not isEffect(df_valid) and not isEffect(df_test):
        return None, None

    # 判断df_train和df_valid样本量是否满足要求
    if isEffect(df_valid) and df_train.shape[0] < lag+gap:
        raise ValueError(
        '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(lag+gap))
    if not isEffect(df_valid) and isEffect(df_test) and \
                                            df_train.shape[0] < lag+gap:
        raise ValueError(
        '根据输入结构判断，训练数据df_train记录数必须大于等于{}！'.format(lag+gap))

    # 构造验证集预测用df
    if isEffect(df_valid):
        n_valid = df_valid.shape[0]
        if n_valid % y_step == 0:
            valid_pre = pd.concat((df_train.iloc[-lag-gap:, :], df_valid),
                                  axis=0).reindex(columns=df_valid.columns)
        else:
            # 处理预测步长与样本量冲突问题
            n_ext = y_step - n_valid % y_step
            if df_train.shape[0] >= lag+gap+y_step:
                logger_show('用训练集数据扩充验证集使得其样本数能被y_step整除！',
                            logger, 'warning')
                df_valid_tmp = pd.concat((df_valid, df_train.iloc[:n_ext, :]),
                                         axis=0)
                valid_pre = pd.concat(
                    (df_train.iloc[-lag-gap:, :], df_valid_tmp), axis=0)
                valid_pre = valid_pre.reindex(columns=df_valid.columns)
            else:
                raise ValueError(
                 '向前多步预测时验证集样本量不能被y_step整除且无法用训练集数据扩充！')
    else:
        valid_pre = None
    # 构造测试集预测用df
    if isEffect(df_test) and isEffect(df_valid):
        n_test = df_test.shape[0]
        tmp = pd.concat((df_train, df_valid), axis=0)
        if n_test % y_step == 0:
            test_pre = pd.concat((tmp.iloc[-lag-gap:, :], df_test), axis=0)
            test_pre = test_pre.reindex(columns=df_test.columns)
        else:
            # 处理预测步长与样本量冲突问题
            n_ext = y_step - n_test % y_step
            if tmp.shape[0] >= lag+gap+y_step:
                logger_show(
                    '用训练集和验证集数据扩充测试集使得其样本数能被y_step整除！',
                    logger, 'warning')
                df_test_tmp = pd.concat((df_test, tmp.iloc[:n_ext, :]), axis=0)
                test_pre = pd.concat(
                    (tmp.iloc[-lag-gap:, :], df_test_tmp), axis=0)
                test_pre = test_pre.reindex(columns=df_test.columns)
            else:
                raise ValueError(
           '向前多步预测时训练集样本量不能被y_step整除且无法用训练集和验证集数据扩充！')
    elif isEffect(df_test) and not isEffect(df_valid):
        n_test = df_test.shape[0]
        if n_test % y_step == 0:
            test_pre = pd.concat((df_train.iloc[-lag-gap:, :], df_test),
                                 axis=0).reindex(columns=df_test.columns)
        else:
            # 处理预测步长与样本量冲突问题
            n_ext = y_step - n_test % y_step
            if df_train.shape[0] >= lag+gap+y_step:
                logger_show('用训练集数据扩充测试集使得其样本数能被y_step整除！',
                            logger, 'warning')
                df_test_tmp = pd.concat((df_test, df_train.iloc[:n_ext, :]),
                                        axis=0)
                test_pre = pd.concat(
                    (df_train.iloc[-lag-gap:, :], df_test_tmp), axis=0)
                test_pre = test_pre.reindex(columns=df_test.columns)
            else:
                raise ValueError(
                 '向前多步预测时验证集样本量不能被y_step整除且无法用训练集数据扩充！')
    else:
        test_pre = None

    # 验证集预测
    if valid_pre is not None:
        valid_pre = forward_predict_X3d_steps(model, funcPred, valid_pre,
                                              ycol, Xcols, lag, gap, y_step,
                                              **kwargs)
        valid_pre = valid_pre.iloc[lag+gap:lag+gap+n_valid, :]
    # 测试集预测
    if test_pre is not None:
        test_pre = forward_predict_X3d_steps(model, funcPred, test_pre, ycol,
                                             Xcols, lag, gap, y_step,
                                             **kwargs)
        test_pre = test_pre.iloc[lag+gap:lag+gap+n_test, :]

    return valid_pre, test_pre


def forward_predict_X3d_steps(model, funcPred, df_pre, ycol, Xcols, lag,
                              gap, y_step, **kwargs):
    '''
    X三维，ycol一维多步向前预测情况
    时间序列向前滚动预测（前面的预测作为后面的输入）
    model为训练好的模型，funcPred为预测函数，接受必要参数model和X，
    也接受可选参数**kwargs，funcPred输出的预测结果长度与输入X的样本数据相同
    df_pre为待预测所用数据，其中ycol为待预测列，除了初始构造输入X必须满足的数据行外，
    后面的数据可以为空值
    Xcols, lag, gap, y_step参数意义同genXy_3d中参数意义，用其构造变量的过程亦完全相同
    返回pd.DataFrame包含原来的数据以及预测结果ycol+'_pre'列
    '''

    if (df_pre.shape[0]-lag-gap) % y_step != 0:
        raise ValueError('必须保证`(df_pre.shape[0]-lag-gap) % y_step == 0`！')

    df_pre[ycol+'_pre'] = df_pre[ycol].copy()
    for k in range(lag+gap, df_pre.shape[0]):
        df_pre.loc[df_pre.index[k], ycol+'_pre'] = np.nan
    if ycol in Xcols:
        ycol_idx = Xcols.index(ycol)
        Xcols_new = Xcols.copy()
        Xcols_new[ycol_idx] = ycol+'_pre'

    for k in range(0, df_pre.shape[0]-lag-gap, y_step):
        X = np.array(df_pre[Xcols_new])[k: k+lag].reshape(1, lag, len(Xcols))

        pred = funcPred(model, X, **kwargs)[0]

        df_pre.loc[df_pre.index[k+lag+gap: k+lag+gap+y_step],
                                                       ycol+'_pre'] = pred

    return df_pre


def genXy_3d(df, ycols, Xcols=None, lag=3, y_step=1, gap=0, y3d=False):
    '''
    多维时间序列构造X和y，输出X为三维：(样本量，lag，特征数)，
    y3d为True时y输出3维度，否则输出2维或1维
    ycols为因变量列名列表，Xcols_series为自变量列名列表，若为None，则设置为ycols
    lag为X期数，y_step为y向前预测期数，gap为间隔期数，即：
        使用前lag天到今天截止的Xcols数据去预测gap天之后共y_step天的ycols
    '''
    Xcols = ycols if Xcols is None else Xcols
    X, y = [], []
    for k in range(df.shape[0]-lag-gap-y_step+1):
        X.append(np.array(df[Xcols])[k: k+lag])
        y.append(np.array(df[ycols])[k+lag+gap: k+lag+gap+y_step])
    X = np.array(X)
    y = np.array(y)
    if not y3d:
        if len(ycols) == 1 and y_step == 1:
            y = y.reshape(-1)
        else:
            y = y.reshape(-1, y_step*len(ycols))
    return X, y
