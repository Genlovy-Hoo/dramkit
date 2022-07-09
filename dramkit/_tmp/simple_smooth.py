# -*- coding: utf-8 -*-

import datetime
import numpy as np


def diff_smooth(series, interval):
    '''
    序列数据平滑处理

    思路：把每个点与上一点的变化值作为一个新的序列，对里边的异常值，
        也就是变化比较离谱的值剃掉，用前后数据的均值填充，

    Parameters
    ----------
    series(pd.Series): 待平滑序列
    interval: ？

    Returns
    -------
    输出为处理后的时间序列

    参考：
    https://zhuanlan.zhihu.com/p/43136171
    '''



    # 间隔为1小时
    wide = interval / 60

    # 差分序列
    dif = series.diff().dropna()

    # 描述性统计得到：min，25%，50%，75%，max值
    td = dif.describe()
    # 定义高点阈值，1.5倍四分位距之外
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    # 定义低点阈值
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

    i = 0
    forbid_index = dif[(dif > high) | (dif < low)].index
    while i < len(forbid_index) - 1:
        # 发现连续多少个点变化幅度过大
        n = 1
        # 异常点的起始索引
        start = forbid_index[i]
        if i+n < len(forbid_index)-1:
            # while forbid_index[i+n] == start + datetime.timedelta(minutes=n):
            while forbid_index[i+n] == start + n:
                n += 1
        i += n - 1
        # 异常点的结束索引
        end = forbid_index[i]
        # 用前后值的中间值均匀填充
        # value = np.linspace(series[start - datetime.timedelta(minutes=wide)], series[end + datetime.timedelta(minutes=wide)], n)
        value = np.linspace(series[start-wide], series[end+wide], n)
        series.iloc[start: end] = value
        i += 1

    return series


if __name__ == '__main__':
    from dramkit import load_csv, plot_series
    
    # 上证50分钟行情------------------------------------------------------------
    fpath = '../../../FinFactory/data/archives/index/joinquant/000016.XSHG_1min.csv'
    his_data = load_csv(fpath).iloc[:-240, :]
    his_data['time'] = his_data['time'].apply(lambda x: x[11:])
    his_data.set_index('time', drop=False, inplace=True)

    df = his_data.iloc[-60:, :].copy()
    # plot_series(df, {'close': '.-b'})

    series = df['close']

    # from scipy.ndimage import gaussian_filter1d

    # df['close_smooth'] = gaussian_filter1d(series, 3)

    # # interval = 60
    # # df['close_smooth'] = diff_smooth(series, interval)

    # plot_series(df.iloc[1:, :], {'close': None, 'close_smooth': None})
