# -*- coding: utf-8 -*-

import PyEMD
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def dec_emds(series, dec_type='EMD'):
    '''
    时间序列信号分解，基于emd的方法（使用EMD-signal库）
    series，时间序列信号数据，pd.Series
    dec_type，采用的分解算法类型，可选'EMD'、'EEMD'和'CEEMDAN'三种
    返回modes，分解之后的成分表，pd.DataFrame格式，每列代表一个成分，
    modes中第一列为最高频成分，倒数第二列为最低频成分，最后一列为分解残差

    '''

    if dec_type == 'EMD':
        method = PyEMD.EMD()
    elif dec_type == 'EEMD':
        method = PyEMD.EEMD()
    elif dec_type == 'CEEMDAN':
        method = PyEMD.CEEMDAN()
    else:
        raise ValueError('分解方法请选择EMD、EEMD和CEEMDAN中的一种！')

    modes = method(np.array(series))
    modes = pd.DataFrame(modes.transpose())
    cols = ['imf_' + str(k) for k in range(1, modes.shape[1]+1)]
    modes.columns = cols
    modes.set_index(series.index, inplace=True)
    modes['dec_res'] = series-modes.transpose().sum()

    return modes


def merge_high_modes(modes, high_num=3):
    '''
    合并时间序列分解后得到的成份数据中的高频成份
    modes，pd.DataFrame,分解结果，高频在前面的列低频在后面的列
    high_num，需要合并的高频成份个数
    返回合并高频成分之后的IMFs，已经删除残差列(dec_res)，每列一个成份，每行一个样本
    '''
    IMFs = modes.copy()
    merge_col_name = 'IMF1_' + str(high_num)
    IMFs[merge_col_name] = IMFs.iloc[:, 0:high_num].transpose().sum()
    IMFs.drop(list(IMFs.columns[0: high_num]), axis=1, inplace=True)
    IMFs.insert(0, merge_col_name, IMFs.pop(merge_col_name))
    if 'dec_res' in IMFs.columns:
        IMFs.drop('dec_res', axis=1, inplace=True)
    return IMFs


def plot_modes(modes, n_xticks=6, figsize=(10, 10)):
    '''
    对分解之后的modes进行绘图查看
    modes，分解之后的数据表，pd.DataFrame格式，每行一个样本，每列一个成份
    n_xticks设置x轴上显示的刻度个数
    '''
    Nplots = modes.shape[1]
    Nsamp = modes.shape[0]
    plt.figure(figsize=figsize)
    for k in range(0, Nplots):
        curr_plot = plt.subplot(Nplots, 1, k+1)
        curr_plot.plot(np.arange(Nsamp), modes.iloc[:,k])
    x_pos = np.linspace(0, Nsamp-1, n_xticks).astype(int)
    plt.xticks(x_pos, list(modes.index[x_pos]), rotation=0)
    plt.tight_layout()
    plt.show()


def SSD(series, lag=10):
    '''奇异值分解时间序列'''

    # 嵌入
    seriesLen = len(series)
    K = seriesLen - lag + 1
    X = np.zeros((lag, K))
    for i in range(K):
        X[:, i] = series[i: i+lag]

    # svd分解，U和sigma已经按升序排序
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT

    # 重组
    rec = np.zeros((lag, seriesLen))
    for i in range(lag):
        for j in range(lag-1):
            for m in range(j+1):
                rec[i, j] += A[i, j-m] * U[m, i]
            rec[i, j] /= (j+1)
        for j in range(lag-1, seriesLen - lag + 1):
            for m in range(lag):
                rec[i, j] += A[i, j-m] * U[m, i]
            rec[i, j] /= lag
        for j in range(seriesLen - lag + 1, seriesLen):
            for m in range(j-seriesLen+lag, lag):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)

    rec = pd.DataFrame(rec).transpose()
    rec.columns = ['imf_' + str(k) for k in range(1, rec.shape[1]+1)]

    return rec


if __name__ == '__main__':
    import time
    strt_tm = time.time()
    from dramkit import load_csv

    fdir = '../_test/'
    fpath = fdir + '510050.SH_daily_qfq.csv'

    series = load_csv(fpath)['close']
    # series = series - series.mean() # 中心化

    series1 = series.iloc[:-1]

    lag = 20 # 嵌入窗口长度

    rec = SSD(series, lag)
    rec1 = SSD(series1, lag)
    rrr = np.sum(rec, axis=1)

    plt.figure(figsize=(12, 9))
    for i in range(10):
        ax = plt.subplot(5, 2, i+1)
        ax.plot(rec.iloc[:, i])
    plt.figure(2)
    plt.plot(series, label='ori')
    plt.plot(rrr, label='rec')
    plt.legend(loc=0)
    plt.show()


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
