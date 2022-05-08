# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime
from dramkit.gentools import get_preval_func_cond


def get_tushare_api(token=None):
    '''根据token获取tushare API接口'''
    if token is None:
        ts_token = ('595253d60e483e240a4924784c23e1e092e7df7193655a66b9ed6862')
    ts.set_token(ts_token)
    ts_api = ts.pro_api()
    return ts_api


def get_trade_date(start_date, end_date, ts_api=None):
    '''
    | 获取从start_date到end_date的交易日索引
    | start_date和end_date格式为'%Y%m%d'
    '''
    if ts_api is None:
        ts_api = get_tushare_api()
    kwargs = {'exchange': 'SSE',
              'start_date': start_date,
              'end_date': end_date,
              'is_open': '1',
              'fields': ['cal_date']}
    df1 = ts_api.trade_cal(**kwargs)
    df2 = pd.DatetimeIndex(df1['cal_date'])
    return df2


def get_aligned_data(data, col, since=None, until=None,
                     ts_api=None, keep_nan=False,
                     drop_headnan=True, notffill=False):
    '''
    | 财报数据对齐到交易日
    | data: 需包含['report_period', 'sid', 'ann_dt']以及因子名称col列
    | since和until: 日期/str，格式'%Y%m%d'，设置返回数据交易日起止日期，
      如不指定，则取data中的起止日期
    | keep_nan: 原始数据中的空值是否保留
    | drop_headnan: 是否删除第一个公告日之前没有数据的部分
    | notffill: 是否不填充交易日空值
    '''

    # 需要保留的数据起止日期
    date0 = data['report_period'].min().strftime('%Y%m%d')
    if since is None:
        since = date0
    if until is None:
        # until = data['ann_dt'].max().strftime('%Y%m%d')
        until = datetime.today().strftime('%Y%m%d')

    # 所有自然日
    naturedays = pd.date_range(date0, until)
    # 所有交易日
    tradedates = get_trade_date(since, until, ts_api=ts_api)

    # 数据转换成矩阵
    df = data.reindex(columns=['ann_dt', 'sid', col])
    if keep_nan:
        df[col] = df[col].fillna(-np.inf)
    # 重复数据以第一条为准
    df.drop_duplicates(subset=['sid', 'ann_dt'],
                       inplace=True)
    df = df.pivot_table(index='ann_dt', columns='sid')
    df.sort_index(ascending=True, inplace=True)

    # 按自然日扩展填充
    df = df.reindex(index=naturedays)
    if notffill:
        df.fillna(np.inf, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # 使用假期最后一天的数据重新填充假期前的最后一个交易日
    tmp = pd.DataFrame({'day': naturedays,
                        'trade': naturedays.isin(tradedates) * 1})
    # 1：表示假期前的最后一个交易日
    # -1：表示假期的最后一天
    tmp['tmp'] = tmp['trade'].diff(-1)
    # 最后一天
    tmp.loc[tmp.index[-1], 'tmp'] = \
                    0 if tmp['day'].iloc[-1] in tradedates else -1
    # 假期前最后一个交易日对应的假期最后一天
    tmp.sort_index(ascending=False, inplace=True)
    tmp['nextlastholiday'] = get_preval_func_cond(tmp, 'day', 'tmp',
                                               lambda x: x == -1)
    tmp.sort_index(ascending=True, inplace=True)
    tmp = tmp[tmp['tmp'] == 1].set_index('day')
    tmp = tmp['nextlastholiday'].to_dict()
    for k, v in tmp.items():
        df.loc[k, :] = df.loc[v, :].values

    # 保留交易日数据
    df = df.reindex(index=tradedates)
    # 矩阵还原为单列
    df = df.stack(dropna=drop_headnan)
    if notffill:
        df.replace(np.inf, np.nan, inplace=True)
    if keep_nan:
        df.replace(-np.inf, np.nan, inplace=True)
    df.index.names = ['cal_date', 'sid']
    df.reset_index(inplace=True)
    df = df.sort_values(['sid', 'cal_date'], ascending=True)

    return df


if __name__ == '__main__':
    import time
    strt_tm = time.time()

    ts_api = get_tushare_api()

    data = pd.read_excel('./test/abs_q_testdata.xlsx')

    # since = '2007-01-01'
    since = None
    # until = '20211106'
    until = None

    df = get_aligned_data(data, 'abs_', since, until,
                          drop_headnan=False)


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
