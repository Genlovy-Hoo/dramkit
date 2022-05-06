# -*- coding: utf-8 -*-

import time
import numpy as np


def cumvar_iter(series, ddof=1):
    '''累计方差计算——迭代'''

    cumvar = np.nan * np.zeros(len(series),)
    for k in range(len(series)):
        cumvar[k] = np.var(series[:k+1], ddof=ddof)

    return cumvar


def cumvar_delta(series, ddof=1):
    '''累计方差计算——增量算法'''

    def delta_var(n0, mean0, var0, n1, mean1, var1, ddof=1):
        '''
        增量方差算法
        '''
        n = n0+n1
        if n0 <= ddof or n1 <= ddof or n <= ddof: # 样本量必须大于自由度
            return np.nan
        fm = n - ddof
        mean = (n0 * mean0 + n1 * mean1) / n
        fz1 = (n0-ddof) * var0 + n0 * (mean - mean0) ** 2
        fz2 = (n1-ddof) * var1 + n1 * (mean - mean1) ** 2
        var = (fz1 + fz2) / fm
        return var

    # 累计均值
    cummean = np.cumsum(series) / np.arange(1, len(series)+1)

    # 累计方差
    cumvar = np.nan * np.ones(len(series),)
    if ddof == 0:
        cumvar[0] = 0
    else:
        for k in range(ddof, ddof+ddof+1):
            cumvar[k] = np.var(series[:k+1], ddof=ddof)
    for k in range(ddof+ddof+1, len(series)):
        var0, mean0, n0 = cumvar[k-ddof-1], cummean[k-ddof-1], k-ddof
        var1 = np.var(series[k-ddof:k+1], ddof=ddof)
        mean1 = np.mean(series[k-ddof:k+1])
        # 增量方差
        cumvar[k] = delta_var(n0, mean0, var0, ddof+1, mean1, var1,
                              ddof=ddof)

    return cumvar


if __name__ == '__main__':

    # 生成一个测试序列
    series = np.random.randint(10, 1000, (50000,))

    start_time = time.time()
    cumvar1 = cumvar_iter(series, ddof=1)
    print('迭代算法用时: {}s.'.format(round(time.time()-start_time, 6)))

    start_time = time.time()
    cumvar2 = cumvar_delta(series, ddof=1)
    print('增量算法用时: {}s.'.format(round(time.time()-start_time, 6)))
