# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import sqrt

from dramkit.gentools import power
from dramkit.gentools import isnull
from dramkit.gentools import cal_pct
from dramkit.gentools import x_div_y
from dramkit.gentools import check_l_allin_l0
from dramkit.gentools import get_update_kwargs
from dramkit.gentools import con_count_ignore
from dramkit.gentools import replace_repeat_func_iter
from dramkit.datetimetools import diff_days_date
from dramkit.logtools.utils_logger import logger_show
from dramkit.plottools.plot_common import plot_series
from dramkit.plottools.plot_common import plot_series_conlabel

#%%
def signal_merge(data, sig1_col, sig2_col, merge_type=1):
    '''
    两个信号合并成一个信号
    
    Parameters
    ----------
    data : pandas.DataFrame
        待处理数据，必须包含 ``sig1_col`` 和 ``sig2_col`` 指定的列
    sig1_col, sig2_col : str
        指定信号列，值为-1表示买（做多），1表示卖（做空）
    merge_type : int
        设置信号合并方式:

        - 1: 两个信号出现任何一个都算有效信号
        - 2: 根据两个信号的持仓量叠加计算交易信号（返回信号不适用反向开仓）
        - 3: 只有两个信号方向相同时才算交易信号（返回信号不适用反向开仓）


    :returns: `pd.Series` - 合并之后的信号
    '''
    df = data.reindex(columns=[sig1_col, sig2_col])
    df.rename(columns={sig1_col: 'sig1', sig2_col: 'sig2'},
              inplace=True)
    if merge_type == 1:
        df['sig'] = df['sig1'] + df['sig2']
        df['sig'] = df['sig'].apply(lambda x: -1 if x < 0 else \
                                    (1 if x > 0 else 0))
        return df['sig']
    elif merge_type == 2:
        df['hold1'] = df['sig1'].replace(0, np.nan)
        df['hold1'] = df['hold1'].fillna(method='ffill').fillna(0)
        df['hold2'] = df['sig2'].replace(0, np.nan)
        df['hold2'] = df['hold2'].fillna(method='ffill').fillna(0)
        df['hold'] = df['hold1'] + df['hold2']
        df['trade'] = df['hold'].diff()
        df['sig'] = df['trade'].apply(lambda x: -1 if x < 0 else \
                                      (1 if x > 0 else 0))
        return df['sig']
    elif merge_type == 3:
        df['hold1'] = df['sig1'].replace(0, np.nan)
        df['hold1'] = df['hold1'].fillna(method='ffill').fillna(0)
        df['hold2'] = df['sig2'].replace(0, np.nan)
        df['hold2'] = df['hold2'].fillna(method='ffill').fillna(0)
        df['hold'] = df['hold1'] + df['hold2']
        df['hold'] = df['hold'].apply(lambda x: 1 if x == 2 else \
                                      (-1 if x == -2 else 0))
        df['trade'] = df['hold'].diff()
        df['sig'] = df['trade'].apply(lambda x: -1 if x < 0 else \
                                      (1 if x > 0 else 0))
        return df['sig']

#%%
def cal_cost_add(hold_vol, hold_cost, add_vol, add_price):
    '''
    | 计算加仓之后的平均持仓成本
    | hold_vol为加仓前持仓量，hold_cost为加仓前平均持仓成本，add_vol为加仓量
    '''
    holdCost = hold_vol * hold_cost
    totCost = holdCost + add_vol * add_price
    return totCost / (hold_vol + add_vol)


def get_mean_cost(trade_records, dirt_col, price_col, vol_col):
    '''
    根据交易记录计算每期持仓成本
    
    Parameters
    ----------
    trade_records : pd.DataFrame
        交易记录数据，必须包含 ``dirt_col`` 、 ``price_col`` 和 `vol_col` 指定的列
    dirt_col : str
        买卖方向列，1为买入（做多），-1为卖出（做空）
    price_col : str
        成交价格列
    vol_col : str
        为成交量列
        

    :returns: `pd.DataFrame` - 在trade_records上增加了'holdVol', 'holdCost', 'meanCost'三列
    '''

    df = trade_records.copy()

    ori_idx = df.index
    df.index = range(0, df.shape[0])

    vol_col_ = vol_col + '_'
    df[vol_col_] = df[dirt_col] * df[vol_col]

    df['holdVol'] = df[vol_col_].cumsum().round(4)
    df.loc[df.index[0], 'holdCost'] = df[price_col].iloc[0] * df[vol_col_].iloc[0]
    df.loc[df.index[0], 'meanCost'] = df[price_col].iloc[0]
    for k in range(1, df.shape[0]):
        holdVol_pre = df['holdVol'].iloc[k-1]
        holdCost_pre = df['holdCost'].iloc[k-1]
        holdVol = df['holdVol'].iloc[k]
        tradeVol = df[vol_col_].iloc[k]
        if tradeVol == 0:
            holdCost, meanCost = holdCost_pre, df['meanCost'].iloc[k-1]
        elif holdVol == 0: # 平仓
            holdCost, meanCost = 0, 0
        elif holdVol_pre >= 0 and holdVol > holdVol_pre: # 买入开仓或加仓
            tradeVal = df[vol_col_].iloc[k] * df[price_col].iloc[k]
            holdCost = holdCost_pre + tradeVal
            meanCost = holdCost / holdVol
        elif holdVol_pre >= 0 and holdVol > 0 and holdVol < holdVol_pre: # 买入减仓
            meanCost = df['meanCost'].iloc[k-1]
            holdCost = meanCost * holdVol
        elif holdVol_pre >= 0 and holdVol < 0: # 买入平仓反向卖出
            meanCost = df[price_col].iloc[k]
            holdCost = holdVol * meanCost
        elif holdVol_pre <= 0 and holdVol < holdVol_pre: # 卖出开仓或加仓
            tradeVal = df[vol_col_].iloc[k] * df[price_col].iloc[k]
            holdCost = holdCost_pre + tradeVal
            meanCost = holdCost / holdVol
        elif holdVol_pre <= 0 and holdVol < 0 and holdVol > holdVol_pre: # 卖出减仓
            meanCost = df['meanCost'].iloc[k-1]
            holdCost = meanCost * holdVol
        elif holdVol_pre <= 0 and holdVol > 0: # 卖出平仓反向买入
            meanCost = df[price_col].iloc[k]
            holdCost = holdVol * meanCost
        df.loc[df.index[k], 'holdCost'] = holdCost
        df.loc[df.index[k], 'meanCost'] = meanCost

    df.index = ori_idx

    return df

#%%
def cal_gain_con_futures(price_open, price_now, n, player,
                         fee=0.1/100, lever=100,
                         n_future2target=0.001):
    '''
    永续合约收益计算，如火币BTC合约
    
    Parameters
    ----------
    price_open : float
        开仓价格
    price_now : float
        现价
    n : int
        数量（张）
    player : str
        做空或做多
    fee : float
        手续费比例
    lever : int
        杠杆
    n_future2target : float
        一份合约对应的标的数量
        
    Returns
    -------
    gain_lever : float
        盈亏金额
    gain_pct : float
        盈亏比例
    '''
    if n == 0:
        return 0, 0
    b_name = ['buyer', 'Buyer', 'b', 'B', 'buy', 'Buy']
    s_name = ['seller', 'Seller', 'seler', 'Seler', 's', 'S', 'sell',
              'Sell', 'sel', 'Sel']
    price_cost_ = price_open * n_future2target / lever
    price_now_ = price_now * n_future2target / lever
    if player in b_name:
        Cost = price_cost_ * n * (1+fee)
        Get = price_now_ * n * (1-fee)
        gain = Get - Cost
        gain_pct = gain / Cost
    elif player in s_name:
        Cost = price_now_ * n * (1+fee)
        Get = price_cost_ * n * (1-fee)
        gain = Get - Cost
        gain_pct = gain / Get
    gain_lever = gain * lever
    return gain_lever, gain_pct


def cal_gain_con_futures2(price_open, price_now, n, player,
                          fee=0.1/100, lever=100):
    '''
    永续合约收益计算，如币安ETH合约
    
    Parameters
    ----------
    price_open : float
        开仓价格
    price_now : float
        现价
    n : int
        数量（标的量）
    player : str
        做空或做多
    fee : float
        手续费比例
    lever : int
        杠杆
        
    Returns
    -------
    gain_lever : float
        盈亏金额
    gain_pct : float
        盈亏比例
    '''
    if n == 0:
        return 0, 0
    b_name = ['buyer', 'Buyer', 'b', 'B', 'buy', 'Buy']
    s_name = ['seller', 'Seller', 'seler', 'Seler', 's', 'S', 'sell',
              'Sell', 'sel', 'Sel']
    price_cost_ = price_open / lever
    price_now_ = price_now / lever
    if player in b_name:
        Cost = price_cost_ * n * (1+fee)
        Get = price_now_ * n * (1-fee)
        gain = Get - Cost
        gain_pct = gain / Cost
    elif player in s_name:
        Cost = price_now_ * n * (1+fee)
        Get = price_cost_ * n * (1-fee)
        gain = Get - Cost
        gain_pct = gain / Get
    gain_lever = gain * lever
    return gain_lever, gain_pct

#%%
def cal_gain_pct_log(price_cost, price, pct_cost0=1):
    '''
    | 计算对数收益率
    | price_cost为成本
    | price为现价
    | pct_cost0为成本price_cost为0时的返回值'''
    if isnull(price_cost) or isnull(price):
        return np.nan
    if price_cost == 0:
        return pct_cost0
    elif price_cost > 0:
        return np.log(price) - np.log(price_cost)
    else:
        raise ValueError('price_cost必须大于等于0！')


def cal_gain_pct(price_cost, price, pct_cost0=1):
    '''
    | 计算百分比收益率
    | price_cost为成本
    | price为现价
    | pct_cost0为成本price_cost为0时的返回值

    Note
    ----
    默认以权利方成本price_cost为正（eg. 买入价为100，则price_cost=100）、
    义务方成本price_cost为负进行计算（eg. 卖出价为100，则price_cost=-100）
    '''
    if isnull(price_cost) or isnull(price):
        return np.nan
    if price_cost > 0:
        return price / price_cost - 1
    elif price_cost < 0:
        return 1 - price / price_cost
    else:
        return pct_cost0


def cal_gain_pcts(price_series, gain_type='pct',
                  pct_cost0=1, logger=None):
    '''
    | 计算资产价值序列price_series(`pd.Series`)每个时间的收益率
    | gain_type:
    |     为'pct'，使用普通百分比收益
    |     为'log'，使用对数收益率（当price_series存在小于等于0时不能使用）
    |     为'dif'，收益率为前后差值（适用于累加净值情况）
    | pct_cost0为当成本为0时收益率的指定值
    '''
    if (price_series <= 0).sum() > 0:
        gain_type = 'pct'
        logger_show('存在小于等于0的值，将用百分比收益率代替对数收益率！',
                    logger, 'warning')
    if gain_type == 'pct':
        df = pd.DataFrame({'price_now': price_series})
        df['price_cost'] = df['price_now'].shift(1)
        df['pct'] = df[['price_cost', 'price_now']].apply(lambda x:
                        cal_gain_pct(x['price_cost'], x['price_now'],
                                     pct_cost0=pct_cost0), axis=1)
        return df['pct']
    elif gain_type == 'log':
        return price_series.apply(np.log).diff()
    elif gain_type == 'dif':
        return price_series.diff()
    else:
        raise ValueError('未识别的`gain_type`，请检查！')

#%%
def cal_beta(values_target, values_base, gain_type='pct', pct_cost0=1):
    '''
    | 计算贝塔系数
    | values_target, values_base分别为目标价值序列和基准价值序列
    | gain_type和pct_cost0同 :func:`dramkit.fintools.utils_gains.cal_gain_pcts` 中的参数
    | 参考：
    | https://www.joinquant.com/help/api/help#api:风险指标
    | https://blog.csdn.net/thfyshz/article/details/83443783
    '''
    values_target = pd.Series(values_target)
    values_base = pd.Series(values_base)
    pcts_target = cal_gain_pcts(values_target, gain_type=gain_type, pct_cost0=pct_cost0)
    pcts_base = cal_gain_pcts(values_base, gain_type=gain_type, pct_cost0=pct_cost0)
    pcts_target = pcts_target.iloc[1:]
    pcts_base = pcts_base.iloc[1:]
    return np.cov(pcts_target, pcts_base)[0][1] / np.var(pcts_base, ddof=1)


def cal_alpha_beta(values_target, values_base, r0=3.0/100, nn=252,
                   gain_type='pct', rtype='exp', pct_cost0=1, logger=None):
    '''
    | 计算alpha和beta系数
    | 参数参考 :func:`cal_beta` 和 :func:`cal_returns_period` 函数
    '''
    r = cal_returns_period(values_target, gain_type=gain_type, rtype=rtype,
                           nn=nn, pct_cost0=pct_cost0, logger=logger)
    r_base = cal_returns_period(values_base, gain_type=gain_type, rtype=rtype,
                                nn=nn, pct_cost0=pct_cost0, logger=logger)
    beta = cal_beta(values_target, values_base,
                    gain_type=gain_type, pct_cost0=pct_cost0)
    return r - (r0 + beta*(r_base-r0)), beta


def cal_alpha_by_beta_and_r(r, r_base, beta, r0=3.0/100):
    '''
    | 根据年化收益以及beta计算alpha
    | r为策略年化收益，r_base为基准年化收益，r0为无风险收益率，beta为策略beta值
    '''
    return r - (r0 + beta*(r_base-r0))

#%%
def cal_return_period_by_gain_pct(gain_pct, n, nn=250, rtype='exp',
                                  gain_pct_type='pct'):
    '''
    给定最终收益率gain_pct，计算周期化收益率
    
    Parameters
    ----------
    gain_pct : float
        给定的最终收益率    
    n : int
        期数
    nn : int
        | 一个完整周期包含的期数，eg.
        |     若price_series周期为日，求年化收益率时nn一般为252（一年的交易日数）
        |     若price_series周期为日，求月度收益率时nn一般为21（一个月的交易日数）
        |     若price_series周期为分钟，求年化收益率时nn一般为252*240（一年的交易分钟数）
    rtype : str
        周期化时采用指数方式'exp'或平均方式'mean'
    gain_pct_type : str
        | 设置最终收益率gain_pct得来的计算方式，可选'pct', 'log'
        | 默认为百分比收益，若为对数收益，则计算周期化收益率时只能采用平均法，不能用指数法

    
    .. hint:: 
        | 百分比收益率：
        |     复利(指数)公式：1 + R = (1 + r) ^ (n / nn) ——> r = (1 + R) ^ (nn / n) - 1
        |     单利(平均)公式：1 + R = 1 + r * (n / nn) ——> r = R * nn / n
        | 对数收益率：
        |     R = r * (n / nn) ——> r = R * nn / n（采用对数收益率计算年化收益只能用平均法）
    
    Returns
    -------
    r : float
        周期化收益率，其周期由nn确定
    
    References
    ----------
    https://zhuanlan.zhihu.com/p/112211063
    '''
    if gain_pct_type in ['log', 'ln', 'lg']:
        rtype = 'mean' # 对数收益率只能采用平均法进行周期化
    if rtype == 'exp':
        r = power(1 + gain_pct, nn / n) - 1
    elif rtype == 'mean':
        r = nn * gain_pct / n
    return r


def cal_ext_return_period_by_gain_pct(gain_pct, gain_pct_base, n,
                                      nn=250, rtype='exp',
                                      gain_pct_type='pct',
                                      ext_type=1):
    '''
    | 给定收益率和基准收益率，计算周期化超额收益率
    | rtype周期化收益率方法，可选'exp'或'mean'或'log'
    | ext_type设置超额收益率计算方式:
    |     若为1，先算各自周期化收益率，再相减
    |     若为2，先相减，再周期化算超额
    |     若为3，先还原两者实际净值，再以相对于基准净值的收益计算周期化超额
    | 其他参数意义同 :func:`cal_return_period_by_gain_pct` 函数

    | 参考：
    | https://xueqiu.com/1930958059/167803003?page=1
    '''
    if rtype == 'log':
        ext_type = 3
    if ext_type == 1:
        p1 = cal_return_period_by_gain_pct(gain_pct, n, nn=nn, rtype=rtype,
                                           gain_pct_type=gain_pct_type)
        p2 = cal_return_period_by_gain_pct(gain_pct_base, n, nn=nn, rtype=rtype,
                                           gain_pct_type=gain_pct_type)
        return p1 - p2
    elif ext_type == 2:
        p = cal_return_period_by_gain_pct(gain_pct-gain_pct_base, n, nn=nn,
                                          rtype=rtype, gain_pct_type=gain_pct_type)
        return p
    if ext_type == 3:
        if gain_pct_type in ['log', 'ln', 'lg']:
            p = np.exp(gain_pct)
            p_base = np.exp(gain_pct_base)
        elif gain_pct_type == 'pct':
            p = 1 + gain_pct
            p_base = 1 + gain_pct_base
        if rtype == 'exp':
            return power(p / p_base, nn / n) - 1
        elif rtype == 'mean':
            return (p / p_base - 1) * nn / n
        elif rtype == 'log':
            return (np.log(p) - np.log(p_base)) * nn / n
    else:
        raise ValueError('未识别的ext_type参数，请检查！')


def cal_ext_return_period(values, values_base, gain_type='pct', rtype='exp',
                          nn=250, pct_cost0=1, ext_type=1, logger=None):
    '''
    | 根据给定价格或价值序列计values和基准序列values_base，算超额收益
    | pct_cost0参考 :func:`cal_gain_pct` 和 :func:`cal_gain_pct_log` 函数
    | 其它参数参考 :func:`cal_ext_return_period_by_gain_pct` 函数
    '''

    values, values_base = np.array(values), np.array(values_base)
    n1, n0 = len(values), len(values_base)
    if n1 != n0:
        raise ValueError('两个序列长度不相等，请检查！')

    if gain_type == 'log':
        if (values[0] <= 0 or values_base[-1] <= 0) or \
                              (values_base[0] <= 0 or values_base[-1] <= 0):
            logger_show('发现开始值或结束值为负，用百分比收益率代替对数收益率！',
                        logger, 'warning')
            p1 = cal_gain_pct(values[0], values[-1], pct_cost0=pct_cost0)
            p0 = cal_gain_pct(values_base[0], values_base[-1], pct_cost0=pct_cost0)
            gain_pct_type = 'pct'
        else:
            p1 = cal_gain_pct_log(values[0], values[-1], pct_cost0=pct_cost0)
            p0 = cal_gain_pct_log(values_base[0], values_base[-1], pct_cost0=pct_cost0)
            rtype = 'mean' # 采用对数收益率计算年化收益只能用平均法
            gain_pct_type = 'log'
    elif gain_type == 'pct':
        p1 = cal_gain_pct(values[0], values[-1], pct_cost0=pct_cost0)
        p0 = cal_gain_pct(values_base[0], values_base[-1], pct_cost0=pct_cost0)
        gain_pct_type = 'pct'
    elif gain_type == 'dif':
        p1 = values[-1] - values[0]
        p1 = values_base[-1] - values_base[0]
        rtype = 'mean'
        gain_pct_type = 'pct'
    else:
        raise ValueError('未识别的`gain_gype`，请检查！')

    extr = cal_ext_return_period_by_gain_pct(p1, p0, n1, nn=nn, rtype=rtype,
                                             gain_pct_type=gain_pct_type,
                                             ext_type=ext_type)

    return extr


def cal_returns_period(price_series, gain_type='pct', rtype='exp',
                       nn=252, pct_cost0=1, logger=None):
    '''
    计算周期化收益率

    Parameters
    ----------
    price_series : pd.Series, np.array, list
        资产价值序列（有负值时不能使用对数收益率）
    gain_type : str
        | 收益率计算方式设置
        |     为'pct'，使用普通百分比收益
        |     为'log'，使用对数收益率（当price_series存在小于等于0时不能使用）
        |     为'dif'，收益率为前后差值（适用于累加净值情况）
    rtype : str
        | 收益率周期化时采用指数方式'exp'或平均方式'mean'
        | （采用对数收益率计算年化收益只能用平均法）
    nn : int
        | 一个完整周期包含的期数，eg.
        |     若price_series周期为日，求年化收益率时nn一般为252（一年的交易日数）
        |     若price_series周期为日，求月度收益率时nn一般为21（一个月的交易日数）
        |     若price_series周期为分钟，求年化收益率时nn一般为252*240（一年的交易分钟数）
    pct_cost0 : float
        成本为0时收益率的指定值，参见 :func:`cal_gain_pct` 和 :func:`cal_gain_pct_log` 函数

    Returns
    -------
    r : float
        周期化收益率，其周期由nn确定
        
    See Also
    --------
    :func:`cal_return_period_by_gain_pct`
    '''

    price_series = np.array(price_series)
    n_ = len(price_series)

    if gain_type == 'log':
        if price_series[0] <= 0 or price_series[-1] <= 0:
            logger_show('发现开始值或结束值为负，用百分比收益率代替对数收益率！',
                        logger, 'warning')
            gain_pct = cal_gain_pct(price_series[0], price_series[-1],
                                    pct_cost0=pct_cost0)
            gain_pct_type = 'pct'
        else:
            gain_pct = cal_gain_pct_log(price_series[0], price_series[-1],
                                        pct_cost0=pct_cost0)
            rtype = 'mean' # 采用对数收益率计算年化收益只能用平均法
            gain_pct_type = 'log'
    elif gain_type == 'pct':
        gain_pct = cal_gain_pct(price_series[0], price_series[-1],
                                pct_cost0=pct_cost0)
        gain_pct_type = 'pct'
    elif gain_type == 'dif':
        gain_pct = price_series[-1] - price_series[0]
        gain_pct_type = 'pct'
        rtype = 'mean'
    else:
        raise ValueError('未识别的`gain_type`，请检查！')

    r = cal_return_period_by_gain_pct(gain_pct, n_, nn=nn, rtype=rtype,
                                      gain_pct_type=gain_pct_type)

    return r


def cal_returns_period_mean(price_series, gain_type='pct', nn=252,
                            pct_cost0=1, logger=None):
    '''
    | 计算周期化收益率，采用收益率直接平均的方法
    | price_series为资产价值序列，pd.Series或list或np.array（有负值时不能使用对数收益率）
    | gain_type和pct_cost0参数参见 :func:`cal_gain_pcts`
    | nn为一个完整周期包含的期数，eg.
    |     若price_series周期为日，求年化收益率时nn一般为252（一年的交易日数）
    |     若price_series周期为日，求月度收益率时nn一般为21（一个月的交易日数）
    |     若price_series周期为分钟，求年化收益率时nn一般为252*240（一年的交易分钟数）
    '''

    price_series = pd.Series(price_series)
    price_series.name = 'series'
    df = pd.DataFrame(price_series)

    # 收益率
    df['gain_pct'] = cal_gain_pcts(price_series, gain_type=gain_type,
                                   pct_cost0=pct_cost0, logger=logger)

    return nn * df['gain_pct'].mean()


def cal_volatility(price_series, gain_type='pct', nn=252,
                   pct_cost0=0, logger=None):
    '''
    | 价格序列price_series的周期化波动率计算
    | price_series为资产价值序列，pd.Series或list或np.array（有负值时不能使用对数收益率）
    | gain_type和pct_cost0参数参见 :func:`cal_gain_pcts`
    | nn为一个完整周期包含的期数，eg.
    |     若price_series周期为日，求年化收益率时nn一般为252（一年的交易日数）
    |     若price_series周期为日，求月度收益率时nn一般为21（一个月的交易日数）
    |     若price_series周期为分钟，求年化收益率时nn一般为252*240（一年的交易分钟数）
    | 返回收益波动率，其周期由nn确定
    | 参考：
    | https://wiki.mbalib.com/wiki/%E5%8E%86%E5%8F%B2%E6%B3%A2%E5%8A%A8%E7%8E%87
    '''

    price_series = pd.Series(price_series)
    price_series.name = 'series'
    df = pd.DataFrame(price_series)

    # 收益率
    df['gain_pct'] = cal_gain_pcts(price_series, gain_type=gain_type,
                                   pct_cost0=pct_cost0, logger=logger)

    # 波动率
    r = df['gain_pct'].std(ddof=1) * sqrt(nn) # ddof为1表示计算标准差时分母为n-1

    return r


def cal_sharpe(values, r=3/100, nn=252, gain_type='pct',
               ann_rtype='exp', pct_cost0=1, logger=None):
    '''
    | 计算夏普比率，先算期望收益和波动率，再算夏普
    | r为无风险收益率
    | 其余参数间 :func:`cal_returns_period` 和 :func:`cal_volatility`
    '''
    return_ann = cal_returns_period(values, gain_type=gain_type,
                                    rtype=ann_rtype, nn=nn,
                                    pct_cost0=pct_cost0, logger=logger)
    volatility = cal_volatility(values, gain_type=gain_type,
                                nn=nn, pct_cost0=pct_cost0,
                                logger=logger)
    sharpe = (return_ann - r) / volatility
    return sharpe


def cal_sharpe2(values, r=3/100, nn=252, gain_type='pct',
                pct_cost0=1, logger=None):
    '''
    计算夏普比率

    Parameters
    ----------
    values : pd.Series
        资产价值序列
    r : float
        无风险收益率
    nn : int
        | 无风险收益率r的周期所包含的values的周期数，eg.
        |     若values周期为日，r为年化无风险收益率时，N一般为252（一年的交易日数）
        |     若values周期为日，r月度无风险收益率时，N一般为21（一个月的交易日数）
        |     若values周期为分钟，r为年化无风险收益率时，N一般为252*240（一年的交易分钟数）
    gain_type : str
        | 收益率计算方式设置
        |     为'pct'，使用普通百分比收益
        |     为'log'，使用对数收益率（当price_series存在小于等于0时不能使用）
        |     为'dif'，收益率为前后差值（适用于累加净值情况）

    References
    ----------
    - https://www.joinquant.com/help/api/help?name=api#风险指标
    - https://www.zhihu.com/question/348938505/answer/1848898074
    - https://blog.csdn.net/thfyshz/article/details/83443783
    - https://www.jianshu.com/p/363aa2dd3441 （夏普计算方法貌似有误）
    '''
    df = pd.DataFrame({'values': values})
    df['gains'] = cal_gain_pcts(df['values'], gain_type=gain_type,
                                pct_cost0=pct_cost0, logger=logger) # 收益率序列
    df['gains_ex'] = df['gains'] - r/nn # 超额收益
    return sqrt(nn) * df['gains_ex'].mean() / df['gains_ex'].std()


def get_maxdown(values, return_idx=True, abs_val=False):
    '''
    最大回撤计算

    Parameters
    ----------
    values : list, np.array, pd.Series
        资产价值序列
    return_idx : bool
        是否返回最大回撤区间起止位置，若为False，则起止位置返回None
    abs_val : bool
        若为True，则计算最大回撤时采用亏损绝对值而不是亏损比率

    Returns
    -------
    maxdown : float
        最大回撤幅度（正值）
    start_end_idxs : tuple
        最大回撤起止位置(start_idx, end_idx)，若return_idx为False，则为(None, None)
    
    References
    ----------
    - https://www.cnblogs.com/xunziji/p/6760019.html
    - https://blog.csdn.net/changyan_123/article/details/80994170
    '''

    n = len(values)
    data = np.array(values)

    if not return_idx:
        maxdown, tmp_max = 0, -np.inf
        for k in range(1, n):
            tmp_max = max(tmp_max, data[k-1])
            if not abs_val:
                maxdown = min(maxdown, data[k] / tmp_max - 1)
            else:
                maxdown = min(maxdown, data[k] - tmp_max)
        start_end_idxs = (None, None)
        return -maxdown, start_end_idxs
    else:
        Cmax, Cmax_idxs = np.zeros(n-1), [0 for _ in range(n-1)]
        tmp_max = -np.inf
        tmp_idx = 0
        for k in range(1, n):
            if data[k-1] > tmp_max:
                tmp_max =  data[k-1]
                tmp_idx = k-1
            Cmax[k-1] = tmp_max
            Cmax_idxs[k-1] = tmp_idx

        maxdown = 0.0
        start_idx, end_idx = 0, 0
        for k in range(1, n):
            if not abs_val:
                tmp = data[k] / Cmax[k-1] - 1
            else:
                tmp = data[k] - Cmax[k-1]
            if tmp < maxdown:
                maxdown = tmp
                start_idx, end_idx = Cmax_idxs[k-1], k
        start_end_idxs = (start_idx, end_idx)
        return -maxdown, start_end_idxs


def get_maxdown_all(values, abs_val=False):
    '''计算区间每个时期的最大回撤'''
    n = len(values)
    data = np.array(values)
    maxdown_all= [0.0]
    maxdown, tmp_max = 0, -np.inf
    for k in range(1, n):
        tmp_max = max(tmp_max, data[k-1])
        if not abs_val:
            maxdown = min(maxdown, data[k] / tmp_max - 1)
            maxdown_all.append(maxdown)
        else:
            maxdown = min(maxdown, data[k] - tmp_max)
            maxdown_all.append(maxdown)
    return np.array(maxdown_all)


def get_maxdown_dy(values, abs_val=False):
    '''计算动态最大回撤（每个时间点之前的最高值到当前的回撤）'''
    data = pd.DataFrame({'values': values})
    data['cummax'] = data['values'].cummax()
    if not abs_val:
        data['dyMaxDown'] = data['values'] / data['cummax'] - 1
    else:
        data['dyMaxDown'] = data['values'] - data['cummax']
    return np.array(data['dyMaxDown'])


def get_maxup(values, return_idx=True, abs_val=False):
    '''
    最大盈利计算（与最大回撤 :func:`dramkit.fintools.utils_gains.get_maxdown` 
    相对应，即做空情况下的最大回撤）
    '''

    n = len(values)
    data = np.array(values)

    if not return_idx:
        maxup, tmp_min = 0, np.inf
        for k in range(1, n):
            tmp_min = min(tmp_min, data[k-1])
            if not abs_val:
                maxup = max(maxup, data[k] / tmp_min - 1)
            else:
                maxup = max(maxup, data[k] - tmp_min)
        return maxup, (None, None)
    else:
        Cmin, Cmin_idxs = np.zeros(n-1), [0 for _ in range(n-1)]
        tmp_min = np.inf
        tmp_idx = 0
        for k in range(1, n):
            if data[k-1] < tmp_min:
                tmp_min =  data[k-1]
                tmp_idx = k-1
            Cmin[k-1] = tmp_min
            Cmin_idxs[k-1] = tmp_idx

        maxup = 0.0
        start_idx, end_idx = 0, 0
        for k in range(1, n):
            if not abs_val:
                tmp = data[k] / Cmin[k-1] - 1
            else:
                tmp = data[k] - Cmin[k-1]
            if tmp > maxup:
                maxup = tmp
                start_idx, end_idx = Cmin_idxs[k-1], k

        return maxup, (start_idx, end_idx)


def get_maxdown_pd(series, abs_val=False):
    '''
    使用pd计算最大回撤计算

    Parameters
    ----------
    series : pd.Series, np.array, list
        资产价值序列
    abs_val : bool
        若为True，则计算最大回撤时采用亏损绝对值而不是亏损比率

    Returns
    -------
    maxdown : float
        最大回撤幅度（正值）
    start_end_idxs : tuple
        最大回撤起止索引：(start_idx, end_idx)
    start_end_iloc : tuple
        最大回撤起止位置（int）：(start_iloc, end_iloc)
    '''

    df = pd.DataFrame(series)
    df.columns = ['val']
    df['idx'] = range(0, df.shape[0])

    df['Cmax'] = df['val'].cummax()
    if not abs_val:
        df['maxdown_now'] = df['val'] / df['Cmax'] - 1
    else:
        df['maxdown_now'] = df['val'] - df['Cmax']

    maxdown = -df['maxdown_now'].min()
    end_iloc = df['maxdown_now'].argmin()
    end_idx = df.index[end_iloc]
    start_idx = df[df['val'] == df.loc[df.index[end_iloc], 'Cmax']].index[0]
    start_iloc = df[df['val'] == df.loc[df.index[end_iloc], 'Cmax']]['idx'][0]

    start_end_idxs = (start_idx, end_idx)
    start_end_iloc = (start_iloc, end_iloc)
    return maxdown, start_end_idxs, start_end_iloc


def cal_n_period_1year(df, col_date='date', n1year=252):
    '''根据交易记录及日期列估计一年的交易期数'''
    days = diff_days_date(df[col_date].max(), df[col_date].min())
    days_trade = days * (n1year / 365)
    nPerTradeDay = df.shape[0] / days_trade
    n_period = int(nPerTradeDay * n1year)
    return n_period

#%%
def get_netval_prod(pct_series):
    '''
    累乘法净值计算，pct_series(`pd.Series`)为收益率序列（注：单位应为%）
    '''
    return (1 + pct_series / 100).cumprod()


def get_netval_sum(pct_series):
    '''
    累加法净值计算，pct_series(`pd.Series`)为收益率序列（注：单位应为%）
    '''
    return (pct_series / 100).cumsum() + 1


def get_gains_act(df_settle):
    '''
    | 根据资金转入转出和资产总值记录计算实际总盈亏%（累计收益/累计投入）
    | df_settle须包含列['转入', '转出', '资产总值']
    | 返回df中包含['累计净流入', '累计投入', '累计盈亏', '实际总盈亏%']这几列
    '''
    df = df_settle.reindex(columns=['转入', '转出', '资产总值'])
    df['净流入'] = df['转入'] - df['转出']
    df['累计净流入'] = df['净流入'].cumsum()
    df['累计盈亏'] = df['资产总值'] - df['累计净流入']
    df['累计净流入_pre'] = df['累计净流入'].shift(1, fill_value=0)
    df['累计投入'] = df['转入'] + df['累计净流入_pre']
    df['实际总盈亏%'] = 100 * df[['累计投入', '累计盈亏']].apply(lambda x:
              x_div_y(x['累计盈亏'], x['累计投入'], v_xy0=0.0, v_y0=1.0), axis=1)
    return df.reindex(columns=['累计净流入', '累计投入', '累计盈亏', '实际总盈亏%'])


def get_fundnet(df_settle, when='before'):
    '''
    用基金净值法根据转入转出和资产总值记录计算净值
    
    Parameters
    ----------
    df_settle : pd.DataFrame
        须包含列['转入', '转出', '资产总值']三列
    when : str
        | 列用于指定当期转入转出计算份额时使用的净值对应的时间:
        |     若为'before'，表示转入转出发生在当天净值结算之前（计算增减份额用前一期净值）
        |     若为'after'，表示转入转出发生在当天净值结算之后（计算增减份额用结算之后的净值）
        |     若为None，当转入大于转出时设置为'before'，当转出大于转入时设置为'after'
        |     若为'when'，在df_settle中通过'when'列指定，'when'列的值只能为'before'或'after'
        
        
    :returns: `pd.DataFrame` - 包含['新增份额', '份额', '净值']三列
    '''
    assert when in [None, 'before', 'after', 'when']
    if when != 'when':
        df = df_settle.reindex(columns=['转入', '转出', '资产总值'])
        if when is None:
            df['when'] = df[['转入', '转出']].apply(lambda x:
                        'before' if x['转入'] >= x['转出'] else 'after', axis=1)
        else:
            df['when'] = when
    else:
        df = df_settle.reindex(columns=['转入', '转出', '资产总值', when])
    assert check_l_allin_l0(df['when'].tolist(), ['before', 'after'])
    ori_index = df.index
    df.reset_index(drop=True, inplace=True)
    df['净流入'] = df['转入'] - df['转出']
    df['份额'] = np.nan
    df['净值'] = np.nan
    for k in range(0, df.shape[0]):
        if k == 0:
            df.loc[df.index[k], '新增份额'] = df.loc[df.index[k], '净流入']
            df.loc[df.index[k], '份额'] = df.loc[df.index[k], '新增份额']
            df.loc[df.index[k], '净值'] = x_div_y(df.loc[df.index[k], '资产总值'],
                                df.loc[df.index[k], '份额'], v_y0=1, v_xy0=1)
        else:
            when = df.loc[df.index[k], 'when']
            if when == 'before':
                df.loc[df.index[k], '新增份额'] = df.loc[df.index[k], '净流入'] / \
                                                     df.loc[df.index[k-1], '净值']
                df.loc[df.index[k], '份额'] = df.loc[df.index[k-1], '份额'] + \
                                                    df.loc[df.index[k], '新增份额']
                df.loc[df.index[k], '净值'] = x_div_y(df.loc[df.index[k], '资产总值'],
                                    df.loc[df.index[k], '份额'], v_y0=1, v_xy0=1)
            else:
                total = df.loc[df.index[k], '资产总值'] - df.loc[df.index[k], '净流入']
                df.loc[df.index[k], '净值'] = x_div_y(total,
                                    df.loc[df.index[k-1], '份额'], v_y0=1, v_xy0=1)
                df.loc[df.index[k], '新增份额'] = df.loc[df.index[k], '净流入'] / \
                                                     df.loc[df.index[k], '净值']
                df.loc[df.index[k], '份额'] = df.loc[df.index[k-1], '份额'] + \
                                              df.loc[df.index[k], '新增份额']
    df.index = ori_index
    return df.reindex(columns=['新增份额', '份额', '净值'])


def get_gains(df_settle, gain_types=['act', 'fundnet'], when=None):
    '''
    | 不同类型的盈亏情况统计
    | gain_types为累计收益计算方法，可选：
    |     ['act'实际总盈亏, 'prod'累乘法, 'sum'累加法, 'fundnet'基金份额净值法]
    | 注意：
    |     累乘法和累加法df_settle须包含'盈亏%'列
    |     实际总盈亏和基金净值法df_settle须包含['转入', '转出', '资产总值']列
    '''

    df_gain = df_settle.copy()
    ori_index = df_gain.index
    df_gain.reset_index(drop=True, inplace=True)

    if 'act' in gain_types:
        cols = ['转入', '转出', '资产总值']
        if any([x not in df_settle.columns for x in cols]):
            raise ValueError('计算实际盈亏要求包含[`转入`, `转出`, `资产总值`]列！')
        df_act = get_gains_act(df_gain)
        df_gain = pd.merge(df_gain, df_act, how='left', left_index=True,
                           right_index=True)

    if 'prod' in gain_types:
        if not '盈亏%' in df_settle.columns:
            raise ValueError('累乘法要求包含`盈亏%`列！')
        df_gain['累乘净值'] = get_netval_prod(df_gain['盈亏%'])

    if 'sum' in gain_types:
        if not '盈亏%' in df_settle.columns:
            raise ValueError('累加法要求包含`盈亏%`列！')
        df_gain['累加净值'] = get_netval_sum(df_gain['盈亏%'])

    if 'fundnet' in gain_types:
        cols = ['转入', '转出', '资产总值']
        if any([x not in df_settle.columns for x in cols]):
            raise ValueError('基金净值法要求包含[`转入`, `转出`, `资产总值`]列！')
        df_net = get_fundnet(df_gain, when=when)
        df_gain = pd.merge(df_gain, df_net, how='left', left_index=True,
                           right_index=True)

    df_gain.index = ori_index

    return df_gain


def plot_gain_act(df_gain, time_col='日期', n=None, **kwargs_plot):
    '''
    | 绘制实际盈亏曲线图
    | df_gain须包含列[time_col, '实际总盈亏%']
    | n设置需要绘制的期数
    | \*\*kwargs_plot为plot_series可接受参数
    '''
    
    n = df_gain.shape[0] if n is None or n < 1 or n > df_gain.shape[0] else n
    df = df_gain.reindex(columns=[time_col, '实际总盈亏%'])
    if n == df_gain.shape[0]:
        df.sort_values(time_col, ascending=True, inplace=True)
        tmp = pd.DataFrame(columns=['日期', '实际总盈亏%'])
        tmp.loc['tmp'] = ['start', 0]
        df = pd.concat((tmp, df), axis=0)
    else:
        df = df.sort_values(time_col, ascending=True).iloc[-n-1:, :]

    df.set_index(time_col, inplace=True)

    if not 'title' in kwargs_plot:
        if n == df_gain.shape[0]:
            kwargs_plot['title'] = '账户实际总盈亏(%)走势'
        else:
            kwargs_plot['title'] = '账户近{}个交易日实际总盈亏(%)走势'.format(n)

    plot_series(df, {'实际总盈亏%': '-ro'}, **kwargs_plot)


def plot_gain_prod(df_gain, time_col='日期', n=None, show_gain=True,
                   **kwargs_plot):
    '''
    | 绘制盈亏净值曲线图
    | df_gain须包含列[time_col, '盈亏%']
    | n设置需要绘制的期数
    | \*\*kwargs为plot_series可接受参数
    '''

    n = df_gain.shape[0] if n is None or n < 1 or n > df_gain.shape[0] else n
    df = df_gain.reindex(columns=[time_col, '盈亏%'])
    if n >= df.shape[0]:
        df.sort_values(time_col, ascending=True, inplace=True)
        tmp = pd.DataFrame(columns=[time_col, '盈亏%'])
        tmp.loc['tmp'] = ['start', 0]
        df = pd.concat((tmp, df), axis=0)
    else:
        df = df.sort_values(time_col, ascending=True).iloc[-n-1:, :]

    df.set_index(time_col, inplace=True)
    df.loc[df.index[0], '盈亏%'] = 0
    df['净值'] = (1 + df['盈亏%'] / 100).cumprod()
    gain_pct = round(100 * (df['净值'].iloc[-1]-1), 2)

    if not 'title' in kwargs_plot:
        if n == df_gain.shape[0]:
            kwargs_plot['title'] = '账户净值曲线\n(收益率: {}%)'.format(gain_pct) \
                              if show_gain else '账户净值曲线'
        else:
             kwargs_plot['title'] = \
                    '账户近{}个交易日净值曲线\n(收益率: {}%)'.format(n, gain_pct) \
                    if show_gain else '账户近{}个交易日净值曲线'.format(n)

    plot_series(df, {'净值': '-ro'}, **kwargs_plot)

#%%
def cal_sig_gains(data, sig_col, sig_type=1, shift_lag=0,
                  col_price='close', col_price_buy='close',
                  col_price_sel='close', settle_after_act=False,
                  func_vol_add='base_1', func_vol_sub='base_1',
                  func_vol_stoploss='hold_1', func_vol_stopgain='hold_1',
                  stop_no_same=True, ignore_no_stop=False,
                  hold_buy_max=None, hold_sel_max=None,
                  limit_min_vol=100, base_money=200000, base_vol=None,
                  init_cash=0.0, fee=1.5/1000, sos_money=1000,
                  max_loss=None, max_gain=None, max_down=None,
                  add_loss_pct=None, add_gain_pct=None,
                  stop_sig_order='both', add_sig_order='offset',
                  force_final0='settle', del_begin0=True,
                  gap_repeat=False, nshow=None, logger=None):
    '''
    统计信号收益情况（没考虑杠杆，适用A股）

    Note
    ----
    仅考虑市场价格为正，不考虑市场价格为负值的极端情况


    .. caution::
        若报错，检查数据中是否存在无效值等

    Parameters
    ----------
    data : pd.DataFrame
        | 行情数据，须包含 ``sig_col``, ``col_price``, ``col_price_buy``,
          ``col_price_sel`` 指定的列。
        | ``col_price`` 为结算价格列；``col_price_buy`` 和 ``col_price_sel``
          分别为做多（买入）和做空（卖出）操作的价格列。
          
          .. note::
              当结算价格的出现时间晚于操作价格时 ``settle_after_act``
              应设置为True，否则 ``settle_after_act`` 设置为False。

        | ``sig_col`` 列为信号列，其值规则应满足：
        | - 当 ``sig_type`` =1时， ``sig_col`` 列的值只能包含-1、1和0，
            其中1为做空（卖出）信号，-1为做多（买入）信号，0为不操作
        | - 当 ``sig_type`` =2时， ``sig_col`` 列的值为正|负整数或0，
            其中正整数表示买入（做多）交易量，负整数表示卖出（做空）交易量，0表示不交易
    func_vol_add : str, function
        | 自定义开仓/加仓操作时的交易量函数，其输入和输出格式应为：
        |     def func_vol_add(base_vol, holdVol, cash, Price):
        |         # Parameters:
        |         #     base_vol：底仓量
        |         #     holdVol：当前持仓量
        |         #     cash: 可用现金
        |         #     Price: 交易价格
        |         # Returns:
        |         #     tradeVol：计划交易量
        |         ......
        |         return tradeVol
        | - 当 ``func_vol_add`` 指定为'base_x'时，使用预定义函数 ``get_AddTradeVol_baseX`` ，
            其交易计划为：开底仓的x倍
        | - 当 ``func_vol_add`` 指定为'hold_x'时，使用预定义函数 ``get_AddTradeVol_holdX`` ，
            其交易计划为：无持仓时开底仓，有持仓时开持仓的x倍
        | - 当 ``func_vol_add`` 指定为'all'时，使用预定义函数 ``get_AddTradeVol_all`` ，
            其交易计划为：以账户当前可用资金为限额开全仓
    func_vol_sub : str, function
        | 自定义平仓/减仓操作时的交易量函数，其输入和输出格式应为：
        |     def func_vol_sub_stop(base_vol, holdVol, cash, Price, holdCost):
        |         # Parameters:
        |         #     base_vol：底仓量
        |         #     holdVol：当前持仓量(正负号表示持仓方向)
        |         #     cash: 可用现金(不包含平|减仓释放的资金)
        |         #     Price: 交易价格
        |         #     holdCost: 当前持仓总成本(用于计算平|减仓释放资金量)
        |         # Returns:
        |         #     tradeVol：计划交易量
        |         ......
        |         return tradeVol
        | - 当指定为'base_x'时，使用预定义函数 ``get_SubStopTradeVol_baseX`` ，
            其交易计划为：减底仓的x倍（若超过了持仓量相当于平仓后反向开仓）
        | - 当指定为'hold_x'时，使用预定义函数 ``get_SubStopTradeVol_holdX`` ，
            其交易计划为：减持仓的x倍（x大于1时相当于平仓后反向开仓）
        | - 当指定为'hold_base_x'时，使用预定义函数 ``get_SubStopTradeVol_holdbaseX`` ，
            其交易计划为：平仓后反向以base_vol的x倍反向开仓
        | - 当指定为'hold_all'时，使用预定义函数 ``get_SubStopTradeVol_holdAll`` ，
            其交易计划为：平仓后以账户当前可用资金（包含平仓释放资金）为限额反向开全仓
    func_vol_stoploss : str, function
        自定义止损操作时的交易量函数，格式同 ``func_vol_sub`` 参数
    func_vol_stopgain : str, function
        自定义止盈操作时的交易量函数，格式同 ``func_vol_sub`` 参数
    stop_no_same : bool
        当止盈止损和操作信号同时出现，是否忽略同向操作信号
        (做多|空止损|盈后是否禁止继续做多|空)
    ignore_no_stop : bool
        当有持仓且没有触及止盈止损条件时是否忽略信号，为True时忽略，为False时不忽略
    hold_buy_max : int, float
        买入(做多)持仓量最大值限制，若信号导致持仓量超限，将被忽略
    hold_sel_max : int, float
        卖出(做空)持仓量最大值限制，若信号导致持仓量超限，将被忽略
    limit_min_vol : int, float
        最少开仓量限制（比如股票至少买100股）
    base_money : float
        开底仓交易限额
    base_vol : int, float
        开底仓交易限量

        .. note::
              同时设置 ``base_money`` 和 ``base_vol`` 时以 ``base_money`` 为准
    init_cash : float
        账户初始资金额
    fee : float
        单向交易综合成本比例（双向收费）
    max_loss : float
        止损比例
    max_gain : float
        止盈比例
    max_down : float
        平仓最大回撤比例
    add_loss_pct : float
        当前价格比上次同向交易价格亏损达到 ``add_loss_pct`` 时加仓，
        加仓方式由 ``func_vol_add`` 决定
    add_gain_pct : float
        当前价格比上次同向交易价格盈利达到 ``add_gain_pct`` 时加仓，
        加仓方式由 ``func_vol_add`` 决定
    stop_sig_order : str
        | - 若为`sig_only`，当止盈止损和操作信号同时出现时，忽略止盈止损信号
        | - 若为`stop_only`，当止盈止损和操作信号同时出现时，忽略操作信号
        | - 若为`stop_first`，当止盈止损和操作信号同时出现时，先考虑止盈止损及反向再开新仓
        | - 若为`sig_first`，当止盈止损和操作信号同时出现时，先考虑操作信号再进行剩余止盈止损及反向
        | - 若为`both`，则止盈止损及反向量和信号交易量同时考虑
    add_sig_order : str
        | - 若为`offset`，当加仓信号与操作信号相反时，两者抵消
        | - 若为`sig_only`，当加仓信号与操作信号相反时，以操作信号为准
        | - 若为`add_only`，当加仓信号与操作信号相反时，以加仓信号为准
    force_final0 : str, bool
        | 最后一个时间强平价格设置：
        | - 若为False，则不强平，按结算价结算账户信息
        | - 若为'trade'，则按col_price_sel或col_price_buy强平
        | - 若为'settle'，则按结算价col_price强平
    sos_money : float
        账户本金亏损完，补资金时保证账户本金最少为 ``sos_money``
    del_begin0 : bool
        是否删除数据中第一个信号之前没有信号的部分数据
    gap_repeat : bool, None, int
        | 重复同向信号处理设置
          (见 :func:`dramkit.gentools.replace_repeat_func_iter` 函数中的 ``gap`` 参数):
        | - 为False时不处理
        | - 为None时重复同向信号只保留第一个
        | - 为整数gap时，重复同向信号每隔gap保留一个

    Returns
    -------
    trade_gain_info : dict
        返回各个收益评价指标
    df : pd.DataFrame
        包含中间过程数据
    
    TODO
    ----
    - 增加根据固定盈亏点位止盈止损/减仓（比如IF赚30个点止盈，亏20个点止损）
    - 增加根据固定盈亏点位加仓（比如IF赚20个点或亏20个点加仓）
    - 止盈止损信号增加动态确定函数(止盈止损或最大回撤百分比不固定，根据持仓时间和盈亏等情况确定)
    - 增加平仓条件设置（比如分段回撤平仓(不同盈利水平不同回撤容忍度)，参数设计成函数）
    - 止损参数和加减仓设计成函数（输入为盈亏比例、最大回撤、盈亏金额、盈亏点位、最近一次交易信息等）
    - 增加浮盈/亏加仓价格确定方式（例如根据距离上次的亏损比例确定）
    - 重写开仓盈亏信息计算函数（先拆单再统计）
    - 正常信号加仓和浮盈/亏加仓方式分开处理（目前是合并一致处理）
    - 固定持仓时间平仓
    - fee买入和卖出分开设置
    - func_vol_add, func_vol_sub, func_vol_stoploss, func_vol_stopgain买入和卖出分开设置
      (考虑股票不能卖空情况，目前可通过设置 hold_sel_max=0控制禁止买空)
    - 添加更多的止盈止损交易量确定方式设置，比如：

        |     1) 分段止盈止损
        |        (eg. 亏10%止损一半仓位，亏20%强平)(需要记录交易过程中的止盈止损历史记录)
        |     2) 止盈止损和加减仓交易量函数参数扩展(加入持仓周期，盈亏比率等参数）
    '''

    def get_base_vol(Price):
        '''计算底仓交易量'''
        if not isnull(base_money):
            Vol_ = limit_min_vol * (base_money // (limit_min_vol * Price * (1+fee)))
            if Vol_ == 0:
                raise ValueError('base_money不够开底仓，请检查设置！')
            return Vol_
        elif isnull(base_vol):
            raise ValueError('base_money和base_vol必须设置一个！')
        else:
            return base_vol

    def get_AddTradeVol_baseX(base_vol, x):
        '''开底仓的x倍'''
        return base_vol * x

    def get_AddTradeVol_holdX(base_vol, holdVol, x):
        '''无持仓则开底仓，有持仓则开持仓的x倍'''
        if abs(holdVol) == 0:
            return base_vol
        return abs(holdVol * x)

    def get_AddTradeVol_all(cash, Price):
        '''以cash为限额计算最大可交易量，Price为交易价格'''
        return limit_min_vol * (cash // (limit_min_vol * Price * (1+fee)))

    def get_SubStopTradeVol_baseX(base_vol, x):
        '''减底仓的x倍（超过持仓即为反向开仓）'''
        return base_vol * x

    def get_SubStopTradeVol_holdX(holdVol, x):
        '''减持仓的x倍（x大于1即为反向开仓）'''
        return abs(holdVol * x)

    def get_SubStopTradeVol_holdbaseX(base_vol, holdVol, x):
        '''平仓后反向开底仓的x倍'''
        return abs(holdVol) + base_vol * x

    def get_SubSellReleaseCash(Vol, Price, Cost):
        '''计算平卖方时释放的资金量'''
        cashput = abs(Vol) * Price * (1+fee)
        gain = abs(Cost) - cashput
        cash_release = abs(Cost) + gain
        return cash_release

    def get_SubStopTradeVol_holdAll(holdVol, cash, Price, holdCost):
        '''平仓后反向开全仓交易量（包含平仓量）'''
        if holdVol >= 0:
            cash_ = holdVol * Price * (1-fee)
            cashall = cash + cash_
            vol_ = limit_min_vol * (cashall // (limit_min_vol * Price * (1+fee)))
        else:
            cash_ = get_SubSellReleaseCash(holdVol, Price, holdCost)
            cashall = cash + cash_
            if cashall <= 0:
                vol_ = 0
            else:
                vol_ = limit_min_vol * (cashall // (limit_min_vol * Price * (1+fee)))
        return abs(holdVol) + vol_

    def get_AddTradeVol(Price, func_vol_add, hold_vol, cash):
        '''开/加仓量计算'''
        base_vol = get_base_vol(Price)
        if isinstance(func_vol_add, str) and 'base' in func_vol_add:
            x = int(float(func_vol_add.split('_')[-1]))
            tradeVol = get_AddTradeVol_baseX(base_vol, x)
        elif isinstance(func_vol_add, str) and 'hold' in func_vol_add:
            x = int(float(func_vol_add.split('_')[-1]))
            tradeVol = get_AddTradeVol_holdX(base_vol, hold_vol, x)
        elif func_vol_add == 'all':
            tradeVol = get_AddTradeVol_all(cash, Price)
        else:
            tradeVol = func_vol_add(base_vol, hold_vol, cash, Price)
        return tradeVol

    def get_SubStopTradeVol(Price, VolF_SubStop, hold_vol, hold_cost, cash):
        '''平/减仓量计算'''
        base_vol = get_base_vol(Price)
        if isinstance(VolF_SubStop, str) and 'hold_base' in VolF_SubStop:
            x = int(float(VolF_SubStop.split('_')[-1]))
            tradeVol = get_SubStopTradeVol_holdbaseX(base_vol, hold_vol, x)
        elif VolF_SubStop == 'hold_all':
            tradeVol = get_SubStopTradeVol_holdAll(hold_vol, cash, Price, hold_cost)
        elif isinstance(VolF_SubStop, str) and 'base' in VolF_SubStop:
            x = int(float(VolF_SubStop.split('_')[-1]))
            tradeVol = get_SubStopTradeVol_baseX(base_vol, x)
        elif isinstance(VolF_SubStop, str) and 'hold' in VolF_SubStop:
            x = int(float(VolF_SubStop.split('_')[-1]))
            tradeVol = get_SubStopTradeVol_holdX(hold_vol, x)
        else:
            tradeVol = VolF_SubStop(base_vol, hold_vol, cash, Price, hold_cost)
        return tradeVol

    def get_tradeVol(sig, act_stop, holdVol_pre, holdCost_pre,
                     buyPrice, selPrice, cash):
        '''
        根据操作信号、止盈止损信号、加仓信号、当前持仓量、持仓成本、交易价格和
        可用资金计算交易方向和计划交易量
        '''
        if sig == 0:
            if holdVol_pre == 0 or act_stop == 0:
                return 0, 0
            else:
                if holdVol_pre > 0: # 做多止损或止盈
                    if act_stop == 0.5: # 做多止损
                        StopTradeVol = get_SubStopTradeVol(selPrice,
                               func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                    else: # 做多止盈
                        StopTradeVol = get_SubStopTradeVol(selPrice,
                               func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                    return 1, StopTradeVol
                elif holdVol_pre < 0: # 做空止损或止盈
                    if act_stop == -0.5: # 做空止损
                        StopTradeVol = get_SubStopTradeVol(buyPrice,
                               func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                    else: # 做空止盈
                        StopTradeVol = get_SubStopTradeVol(buyPrice,
                               func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                    return -1, StopTradeVol
        elif holdVol_pre == 0:
            if sig_type == 1:
                tradePrice = buyPrice if sig == -1 else selPrice
                tradeVol = get_AddTradeVol(tradePrice, func_vol_add,
                                                             holdVol_pre, cash)
                return sig, tradeVol
            elif sig_type == 2:
                dirt = -1 if sig > 0 else 1
                return dirt, abs(sig)
        elif holdVol_pre > 0: # 持有做多仓位
            if (sig_type == 1 and sig == 1) or (sig_type == 2 and sig < 0):
                if act_stop == 0:
                    if not ignore_no_stop: # 正常减/平做多仓位
                        if sig_type == 1:
                            selVol = get_SubStopTradeVol(selPrice, func_vol_sub,
                                               holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            selVol = abs(sig)
                        return 1, selVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 做多仓位止盈止损信号与做空信号结合
                    if stop_sig_order == 'sig_only':
                        if sig_type == 1:
                            selVol = get_SubStopTradeVol(selPrice, func_vol_sub,
                                               holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            selVol = abs(sig)
                        return 1, selVol
                    elif stop_sig_order == 'stop_only':
                        if act_stop == 0.5:
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else:
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        return 1, StopTradeVol
                    elif stop_sig_order == 'stop_first':
                        if act_stop == 0.5: # 先止损后再开做空仓位
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else: # 先止盈后再开做空仓位
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        if sig_type == 1:
                            cash_ = cash + holdVol_pre * selPrice * (1-fee)
                            selVol = get_AddTradeVol(selPrice, func_vol_add, 0, cash_)
                        elif sig_type == 2:
                            selVol = abs(sig)
                        return 1, max(StopTradeVol, selVol+holdVol_pre)
                    elif stop_sig_order == 'sig_first':
                        if sig_type == 1:
                            selVol = get_SubStopTradeVol(selPrice, func_vol_sub,
                                               holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            selVol = abs(sig)
                        LeftVol = holdVol_pre - selVol # 剩余做多仓位
                        if LeftVol <= 0: # 信号卖出量已经超过持仓量
                            return 1, selVol
                        cash_ = cash + selVol * selPrice * (1-fee)
                        holdCost_pre_ = holdCost_pre * (LeftVol / holdVol_pre)
                        if act_stop == 0.5: # 需要止损剩余仓位
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                  func_vol_stoploss, LeftVol, holdCost_pre_, cash_)
                        else: # 需要止盈剩余仓位
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                  func_vol_stopgain, LeftVol, holdCost_pre_, cash_)
                        return 1, StopTradeVol+selVol
                    elif stop_sig_order == 'both':
                        if sig_type == 1:
                            selVol = get_SubStopTradeVol(selPrice, func_vol_sub,
                                               holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            selVol = abs(sig)
                        if act_stop == 0.5:
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else:
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        return 1, max(StopTradeVol, selVol)
                    else:
                        raise ValueError('`stop_sig_order`参数设置错误！')
            elif (sig_type == 1 and sig == -1) or (sig_type == 2 and sig > 0):
                if act_stop == 0:
                    if not ignore_no_stop: # 正常加做多仓位
                        if sig_type == 1:
                            buyVol = get_AddTradeVol(buyPrice, func_vol_add,
                                                     holdVol_pre, cash)
                        elif sig_type == 2:
                            buyVol = abs(sig)
                        return -1, buyVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 做多仓位止盈止损信号与做多信号结合
                    if stop_no_same or stop_sig_order == 'stop_only': # 做多止盈止损后禁止做多
                        if act_stop == 0.5: # 止损
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else: # 止盈
                            StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        return 1, StopTradeVol
                    if sig_type == 1:
                        buyVol = get_AddTradeVol(buyPrice, func_vol_add,
                                                 holdVol_pre, cash)
                    elif sig_type == 2:
                        buyVol = abs(sig)
                    if stop_sig_order == 'sig_only':
                        return -1, buyVol
                    if act_stop == 0.5:
                        StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                    else:
                        StopTradeVol = get_SubStopTradeVol(selPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                    if buyVol == StopTradeVol:
                        return 0, 0
                    elif buyVol > StopTradeVol:
                        return -1, buyVol-StopTradeVol
                    else:
                        return 1, StopTradeVol-buyVol
        elif holdVol_pre < 0: # 持有做空仓位
            if (sig_type == 1 and sig == 1) or (sig_type == 2 and sig < 0):
                if act_stop == 0:
                    if not ignore_no_stop: # 正常加做空仓位
                        if sig_type == 1:
                            selVol = get_AddTradeVol(selPrice, func_vol_add,
                                                     holdVol_pre, cash)
                        elif sig_type == 2:
                            selVol = abs(sig)
                        return 1, selVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 做空仓位止盈止损信号与做空信号结合
                    if stop_no_same or stop_sig_order == 'stop_only': # 做空止盈止损后禁止做空
                        if act_stop == 0.5: # 止损
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else: # 止盈
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        return -1, StopTradeVol
                    if sig_type == 1:
                        selVol = get_AddTradeVol(selPrice, func_vol_add,
                                                 holdVol_pre, cash)
                    elif sig_type == 2:
                        selVol = abs(sig)
                    if stop_sig_order == 'sig_only':
                        return 1, selVol
                    if act_stop == -0.5:
                        StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                    else:
                        StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                    if selVol == StopTradeVol:
                        return 0, 0
                    elif selVol > StopTradeVol:
                        return 1, selVol-StopTradeVol
                    else:
                        return -1, StopTradeVol-selVol
            elif (sig_type == 1 and sig == -1) or (sig_type == 2 and sig > 0):
                if act_stop == 0:
                    if not ignore_no_stop: # 正常减/平做空仓位
                        if sig_type == 1:
                            buyVol = get_SubStopTradeVol(buyPrice, func_vol_sub,
                                               holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            buyVol = abs(sig)
                        return -1, buyVol
                    else: # 不触及止盈止损时忽略信号（不操作）
                        return 0, 0
                else: # 做空仓位止盈止损信号与做多信号结合
                    if stop_sig_order == 'sig_only':
                        if sig_type == 1:
                            buyVol = get_SubStopTradeVol(buyPrice, func_vol_sub,
                                              holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            buyVol = abs(sig)
                        return -1, buyVol
                    elif stop_sig_order == 'stop_only':
                        if act_stop == -0.5:
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else:
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        return  -1, StopTradeVol
                    elif stop_sig_order == 'stop_first':
                        if act_stop == -0.5: # 先止损再开做多仓位
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else: # 先止盈再开做多仓位
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        if sig_type == 1:
                            cash_ = cash + get_SubSellReleaseCash(holdVol_pre,
                                                        buyPrice, holdCost_pre)
                            buyVol = get_AddTradeVol(buyPrice, func_vol_add,
                                                                      0, cash_)
                        elif sig_type == 2:
                            buyVol = abs(sig)
                        return -1, max(StopTradeVol, buyVol+abs(holdVol_pre))
                    elif stop_sig_order == 'sig_first':
                        if sig_type == 1:
                            buyVol = get_SubStopTradeVol(buyPrice, func_vol_sub,
                                              holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            buyVol = abs(sig)
                        LeftVol = holdVol_pre + buyVol # 剩余做空仓位
                        if LeftVol >= 0: # 信号买入量已经超过持仓量
                            return -1, buyVol
                        cash_ = cash + get_SubSellReleaseCash(buyVol, buyPrice,
                                         holdCost_pre * (buyVol / holdVol_pre))
                        holdCost_pre_ = holdCost_pre * (LeftVol / holdVol_pre)
                        if act_stop == -0.5: # 需要止损剩余仓位
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stoploss, LeftVol, holdCost_pre_, cash_)
                        else: # 需要止盈剩余仓位
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stopgain, LeftVol, holdCost_pre_, cash_)
                        return -1, StopTradeVol+buyVol
                    elif stop_sig_order == 'both':
                        if sig_type == 1:
                            buyVol = get_SubStopTradeVol(buyPrice, func_vol_sub,
                                               holdVol_pre, holdCost_pre, cash)
                        elif sig_type == 2:
                            buyVol = abs(sig)
                        if act_stop == -0.5:
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stoploss, holdVol_pre, holdCost_pre, cash)
                        else:
                            StopTradeVol = get_SubStopTradeVol(buyPrice,
                                func_vol_stopgain, holdVol_pre, holdCost_pre, cash)
                        return -1, max(StopTradeVol, buyVol)
                    else:
                        raise ValueError('`stop_sig_order`参数设置错误！')

    def buy_act(df, k, buy_price, buy_vol, hold_vol_pre, hold_cost_pre,
                hold_cash_pre):
        '''买入操作记录'''
        df.loc[df.index[k], 'act_price'] = buy_price
        df.loc[df.index[k], 'buyVol'] = buy_vol
        df.loc[df.index[k], 'holdVol'] = buy_vol + hold_vol_pre
        cashPut = buy_vol * buy_price * (1+fee)
        df.loc[df.index[k], 'cashPut'] = cashPut
        if hold_vol_pre >= 0: # 做多加仓或开仓
            df.loc[df.index[k], 'holdCost'] = hold_cost_pre + cashPut
            if cashPut >= hold_cash_pre:
                cashNeed = cashPut - hold_cash_pre
                holdCash = 0
            else:
                cashNeed = 0
                holdCash = hold_cash_pre - cashPut
        else:
            # 之前持有的平均成本
            hold_cost_mean_pre = hold_cost_pre / hold_vol_pre
            if buy_vol <= abs(hold_vol_pre): # 减或平做空仓位
                if buy_vol == abs(hold_vol_pre):
                    df.loc[df.index[k], 'holdCost'] = 0
                else:
                    df.loc[df.index[k], 'holdCost'] = hold_cost_mean_pre * \
                                                      (buy_vol + hold_vol_pre)
                    # df.loc[df.index[k], 'holdCost'] = hold_cost_pre + cashPut
                # cashGet_pre为空单开仓时的成本金
                # （注：默认空单成本为负值，没考虑成本为正值情况）
                cashGet_pre = buy_vol * hold_cost_mean_pre
                gain = cashGet_pre - cashPut # 空单盈亏
                df.loc[df.index[k], 'act_gain_Cost'] = -cashGet_pre
                df.loc[df.index[k], 'act_gain_Val'] = -cashPut
                # 若空单赚钱，盈利和占用资金转化为现金
                # 若空单亏钱，平仓可能需要补资金，占用资金除去亏损剩余部分转化为现金
                if gain >= 0:
                    cashNeed = 0
                    holdCash = (cashGet_pre + gain) + hold_cash_pre
                elif gain < 0:
                    cashNeed_ = abs(gain) - cashGet_pre
                    if cashNeed_ >= hold_cash_pre:
                        cashNeed = cashNeed_ - hold_cash_pre
                        holdCash = 0
                    else:
                        cashNeed = 0
                        holdCash = hold_cash_pre + (cashGet_pre + gain)
            elif buy_vol > abs(hold_vol_pre): # 平做空仓位后反向开做多仓位
                # buyNeed_反向开多单需要的资金
                buyNeed_ = (buy_vol + hold_vol_pre) * buy_price * (1+fee)
                df.loc[df.index[k], 'holdCost'] = buyNeed_
                # gain空单盈亏
                val_sub = hold_vol_pre*buy_price*(1+fee)
                gain = val_sub - hold_cost_pre
                cash_ = hold_cash_pre + (gain-hold_cost_pre)
                if buyNeed_ >= cash_:
                    cashNeed = buyNeed_ - cash_
                    holdCash = 0
                else:
                    cashNeed = 0
                    holdCash = cash_ - buyNeed_
                df.loc[df.index[k], 'act_gain_Cost'] = hold_cost_pre
                df.loc[df.index[k], 'act_gain_Val'] = val_sub
            df.loc[df.index[k], 'act_gain'] = gain
            if buy_price*(1+fee) <= hold_cost_mean_pre:
                df.loc[df.index[k], 'act_gain_label'] = 1
            elif buy_price*(1+fee) > hold_cost_mean_pre:
                df.loc[df.index[k], 'act_gain_label'] = -1
        if k == 0:
            df.loc[df.index[k], 'cashNeed'] = max(init_cash,
                                                  cashNeed+init_cash)
        else:
            df.loc[df.index[k], 'cashNeed'] = cashNeed
        df.loc[df.index[k], 'holdCash'] = holdCash
        return df

    def sel_act(df, k, sel_price, sel_vol, hold_vol_pre, hold_cost_pre,
                hold_cash_pre):
        '''卖出操作记录'''
        df.loc[df.index[k], 'act_price'] = sel_price
        df.loc[df.index[k], 'selVol'] = sel_vol
        df.loc[df.index[k], 'holdVol'] = hold_vol_pre - sel_vol
        cashGet = sel_vol * sel_price * (1-fee)
        df.loc[df.index[k], 'cashGet'] = cashGet
        if hold_vol_pre <= 0: # 做空加仓或开仓
            df.loc[df.index[k], 'holdCost'] = hold_cost_pre - cashGet
            cashNeed_ = sel_vol * sel_price * (1+fee)
        else:
            # 之前持有的平均成本
            hold_cost_mean_pre = hold_cost_pre / hold_vol_pre
            if sel_vol <= hold_vol_pre: # 减或平做多仓位
                if sel_vol == hold_vol_pre:
                    df.loc[df.index[k], 'holdCost'] = 0
                else:
                    df.loc[df.index[k], 'holdCost'] = hold_cost_mean_pre * \
                                                      (hold_vol_pre - sel_vol)
                    # df.loc[df.index[k], 'holdCost'] = hold_cost_pre - cashGet
                cashNeed_ = -cashGet
                cashPut_pre = hold_cost_mean_pre * sel_vol
                gain = cashGet - cashPut_pre
                df.loc[df.index[k], 'act_gain_Cost'] = cashPut_pre
                df.loc[df.index[k], 'act_gain_Val'] = cashGet
            elif sel_vol > hold_vol_pre: # 平做多仓位后反向开做空仓位
                df.loc[df.index[k], 'holdCost'] = \
                                (hold_vol_pre-sel_vol) * sel_price * (1-fee)
                cashGet_pre = hold_vol_pre * sel_price * (1-fee)
                cashNeed_ = (sel_vol-hold_vol_pre) * sel_price * (1+fee) \
                            - cashGet_pre
                gain = cashGet_pre - hold_cost_pre
                df.loc[df.index[k], 'act_gain_Cost'] = hold_cost_pre
                df.loc[df.index[k], 'act_gain_Val'] = cashGet_pre
            df.loc[df.index[k], 'act_gain'] = gain
            if sel_price*(1-fee) >= hold_cost_mean_pre:
                df.loc[df.index[k], 'act_gain_label'] = 1
            elif sel_price*(1-fee) < hold_cost_mean_pre:
                df.loc[df.index[k], 'act_gain_label'] = -1
        if cashNeed_ >= hold_cash_pre:
            if k == 0:
                df.loc[df.index[k], 'cashNeed'] = max(init_cash,
                                        init_cash + cashNeed_ - hold_cash_pre)
            else:
                df.loc[df.index[k], 'cashNeed'] = cashNeed_ - hold_cash_pre
            df.loc[df.index[k], 'holdCash'] = 0
        else:
            if k == 0:
                df.loc[df.index[k], 'cashNeed'] = init_cash
            else:
                df.loc[df.index[k], 'cashNeed'] = 0
            df.loc[df.index[k], 'holdCash'] = hold_cash_pre - cashNeed_
        return df

    if sig_type == 1:
        act_types = list(data[sig_col].unique())
        if not check_l_allin_l0(act_types, [0, 1, -1]):
            raise ValueError('data.{}列的值只能是0或1或-1！'.format(sig_col))

    assert (data[sig_col] == 0).sum() < data.shape[0], '{}信号列全为0！'.format(sig_col)

    if force_final0:
        if force_final0 not in ['trade', 'settle']:
            raise ValueError('请设置`force_final0`为False或`trade`或`settle`')

    cols = list(set([sig_col, col_price, col_price_buy, col_price_sel]))
    df = data.reindex(columns=cols)

    for col in cols:
        if df[col].isna().sum() > 0:
            raise ValueError('{}列存在无效值，请检查！'.format(col))

    df[sig_col] = df[sig_col].shift(shift_lag)
    df[sig_col] = df[sig_col].fillna(0)

    if del_begin0:
        k = 0
        while k < df.shape[0] and df[sig_col].iloc[k] == 0:
            k += 1
        df = df.iloc[k:, :].copy()

    if gap_repeat != False:
        df[sig_col] = replace_repeat_func_iter(df[sig_col],
                                            lambda x: x > 0,
                                            lambda x: 0,
                                            gap=gap_repeat)
        df[sig_col] = replace_repeat_func_iter(df[sig_col],
                                            lambda x: x < 0,
                                            lambda x: 0,
                                            gap=gap_repeat)

    # 新增一条记录用于最终强平（避免最后一条记录有信号时与强平混淆）
    iend = '{}_'.format(df.index[-1])
    if df[sig_col].iloc[-1] != 0:
        logger_show('将新增一条记录处理最终结算|强平（会略微影响收益评价指标）。',
                    logger, level='warn')
        df.loc[iend, :] = df.iloc[-1, :]
        df.loc[iend, sig_col] = 0

    ori_index = df.index
    df.reset_index(drop=True, inplace=True)

    df['buyVol'] = 0 # 做多（买入）量
    df['selVol'] = 0 # 做空（卖出）量
    df['holdVol'] = 0 # 持仓量（交易完成后）
    df['holdVal'] = 0 # 持仓价值（交易完成后）
    df['cashPut'] = 0 # 现金流出（买入做多算手续费后成本资金）
    df['cashGet'] = 0 # 现金流入（卖出做空算手续费后成本资金）
    df['cashNeed'] = 0 # 交易时账户需要转入的资金（当笔交易）
    df.loc[df.index[0], 'cashNeed'] = init_cash
    df['holdCash'] = 0 # 账户持有现金（交易完成后）

    df['holdCost'] = 0 # 现有持仓总成本（交易完成后）
    df['holdCost_mean'] = 0 # 现有持仓平均成本（交易完成后）
    df['holdPreGainPct'] = 0 # 持仓盈亏（交易完成前）
    df['holdPreGainPctMax'] = 0 # 持仓达到过的最高收益（交易完成前）
    df['holdPreLossPctMax'] = 0 # 持仓达到过的最大亏损（交易完成前）
    df['holdPreMaxDown'] = 0 # 持仓最大回撤（交易完成前）
    df['act_stop'] = 0 # 止盈止损标注（0.5多止损，1.5多止盈，-0.5空止损，-1.5空止盈）
    df['act'] = 0 # 实际操作（1做空，-1做多）（用于信号被过滤时进行更正）
    df['act_price'] = np.nan # 交易价格
    df['act_gain'] = 0 # 平|减仓盈亏
    df['act_gain_Cost'] = 0 # 减|平仓位的总成本
    df['act_gain_Val'] = 0 # 减|平仓位的交易额
    df['act_gain_label'] = 0 # 若操作为平仓或减仓，记录盈亏标志（1为盈利，-1为亏损）
    df['holdGainPct'] = 0 # 现有持仓盈亏（交易完成后）

    last_act = 0 # 上一个操作类型
    last_acted = np.nan # 上一个操作类型，只有操作才更新
    last_actPrice = np.nan # 上一个操作类型，只有操作才更新
    for k in range(0, df.shape[0]):
        if nshow and k % nshow == 0:
            logger_show('simTrading: {} / {}, {} ...'.format(
                        k, df.shape[0], ori_index[k]),
                        logger)

        # 交易前持仓量
        if k == 0:
            holdVol_pre = 0
            holdCost_pre = 0
            act_stop = 0
            act_add = 0
            holdPreGainPct = 0
            holdPreGainPctMax = 0
            holdPreLossPctMax = 0
            holdPreMaxDown = 0
            holdCash_pre = init_cash
            Price_settle = np.nan
        else:
            holdVol_pre = df.loc[df.index[k-1], 'holdVol']
            holdCost_pre = df.loc[df.index[k-1], 'holdCost']
            holdCash_pre = df.loc[df.index[k-1], 'holdCash']
            if holdVol_pre == 0:
                act_stop = 0
                holdPreGainPct = 0
                holdPreGainPctMax = 0
                holdPreLossPctMax = 0
                holdPreMaxDown = 0
            else:
                # 检查止盈止损及加仓条件是否触及
                if settle_after_act:
                    Price_settle = df.loc[df.index[k-1], col_price]
                else:
                    Price_settle = df.loc[df.index[k], col_price]
                if holdVol_pre > 0:
                    holdVal_pre = holdVol_pre * Price_settle * (1-fee)
                elif holdVol_pre < 0:
                    holdVal_pre = holdVol_pre * Price_settle * (1+fee)
                holdPreGainPct = cal_gain_pct(holdCost_pre, holdVal_pre, pct_cost0=0)
                # 若前一次有操作，则计算持仓盈利和回撤（交易前）须重新计算
                if last_act == 0:
                    holdPreGainPctMax = max(holdPreGainPct,
                                    df.loc[df.index[k-1], 'holdPreGainPctMax'])
                    holdPreLossPctMax = min(holdPreGainPct,
                                    df.loc[df.index[k-1], 'holdPreLossPctMax'])
                else:
                    holdPreGainPctMax = max(holdPreGainPct, 0)
                    holdPreLossPctMax = min(holdPreGainPct, 0)
                # 最大回撤
                # holdPreMaxDown = holdPreGainPctMax - holdPreGainPct
                if holdCost_pre > 0:
                    holdValMax_pre = holdCost_pre * (1 + holdPreGainPctMax)
                elif holdCost_pre < 0:
                    holdValMax_pre = holdCost_pre * (1 - holdPreGainPctMax)
                holdPreMaxDown = abs(cal_gain_pct(holdValMax_pre, holdVal_pre,
                                                  pct_cost0=0))

                # 没有止盈止损
                if isnull(max_loss) and isnull(max_gain) and isnull(max_down):
                    act_stop = 0
                # 固定比率止盈止损
                elif not isnull(max_loss) or not isnull(max_gain):
                    # 止损
                    if not isnull(max_loss) and holdPreGainPct <= -max_loss:
                        if holdCost_pre > 0:
                            act_stop = 0.5 # 做多止损
                        elif holdCost_pre < 0:
                            act_stop = -0.5 # 做空止损
                        else:
                            act_stop = 0
                    # 止盈
                    elif not isnull(max_gain) and holdPreGainPct >= max_gain:
                        if holdCost_pre > 0:
                            act_stop = 1.5 # 做多止盈
                        elif holdCost_pre < 0:
                            act_stop = -1.5 # 做空止盈
                        else:
                            act_stop = 0
                    else:
                        act_stop = 0
                # 最大回撤平仓
                elif not isnull(max_down):
                    if holdPreMaxDown < max_down:
                        act_stop = 0
                    else:
                        if holdCost_pre > 0:
                            # act_stop = 0.5 # 做多平仓
                            if holdPreGainPct < 0:
                                act_stop = 0.5 # 做多止损
                            elif holdPreGainPct > 0:
                                act_stop = 1.5 # 做多止盈
                            else:
                                act_stop = 0
                        elif holdCost_pre < 0:
                            # act_stop = -0.5 # 做空平仓
                            if holdPreGainPct < 0:
                                act_stop = -0.5 # 做空止损
                            elif holdPreGainPct > 0:
                                act_stop = -1.5 # 做空止盈
                            else:
                                act_stop = 0
                        else:
                            act_stop = 0

                # 没有加仓
                if (isnull(add_loss_pct) and isnull(add_gain_pct)) \
                                        or holdVol_pre == 0 or last_acted == 0:
                    act_add = 0
                else:
                    # 比上次同向操作时价格涨跌幅
                    pct_lastacted = cal_pct(last_actPrice, Price_settle)
                    # 浮盈加仓|浮亏加仓
                    if holdVol_pre > 0 and last_acted == -1 and \
                          not isnull(add_gain_pct) and pct_lastacted >= add_gain_pct:
                        act_add = -1 # 做多浮盈加仓
                    elif holdVol_pre < 0 and last_acted == 1 and \
                          not isnull(add_gain_pct) and pct_lastacted <- add_gain_pct:
                        act_add = 1 # 做空浮盈加仓
                    elif holdVol_pre > 0 and last_acted == -1 and \
                          not isnull(add_loss_pct) and pct_lastacted <= -add_loss_pct:
                        act_add = -1 # 做多浮亏加仓
                    elif holdVol_pre < 0 and last_acted == 1 and \
                          not isnull(add_loss_pct) and pct_lastacted >= add_loss_pct:
                        act_add = 1 # 做空浮亏加仓
                    else:
                        act_add = 0

        df.loc[df.index[k], 'act_stop'] = act_stop
        df.loc[df.index[k], 'holdPreGainPct'] = holdPreGainPct
        df.loc[df.index[k], 'holdPreGainPctMax'] = holdPreGainPctMax
        df.loc[df.index[k], 'holdPreLossPctMax'] = holdPreLossPctMax
        df.loc[df.index[k], 'holdPreMaxDown'] = holdPreMaxDown

        buyPrice = df.loc[df.index[k], col_price_buy]
        selPrice = df.loc[df.index[k], col_price_sel]
        sig = df.loc[df.index[k], sig_col] # 操作信号

        if sig == 0:
            sig = act_add
        elif sig + act_add == 0:
            if add_sig_order == 'offset':
                sig = 0
            elif add_sig_order == 'add_only':
                sig = act_add
            elif add_sig_order == 'sig_only':
                sig == sig
            else:
                raise ValueError('`add_sig_order`参数设置错误！')

        # 确定交易计划
        if k == df.shape[0]-1 and force_final0:
            if holdVol_pre > 0:
                act, tradeVol  = 1, holdVol_pre # 强平多仓
                last_force = 'sel'
                if force_final0 == 'settle':
                    selPrice = df.loc[df.index[k], col_price]
            elif holdVol_pre < 0:
                act, tradeVol = -1, abs(holdVol_pre) # 强平空仓
                last_force = 'buy'
                if force_final0 == 'settle':
                    buyPrice = df.loc[df.index[k], col_price]
            else:
                act, tradeVol = 0, 0
                last_force = None
        else:
            act, tradeVol = get_tradeVol(sig, act_stop, holdVol_pre,
                                holdCost_pre, buyPrice, selPrice, holdCash_pre)

        # 检查交易后是否会导致持仓量超限，更正交易量
        if act == -1:
            if not isnull(hold_buy_max):
                if holdVol_pre >= hold_buy_max:
                    act, tradeVol = 0, 0
                elif holdVol_pre + tradeVol > hold_buy_max:
                    act, tradeVol = -1, hold_buy_max-holdVol_pre
        elif act == 1:
            if not isnull(hold_sel_max):
                if -holdVol_pre >= hold_sel_max:
                    act, tradeVol = 0, 0
                elif -holdVol_pre + tradeVol > hold_sel_max:
                    act, tradeVol = 1, hold_sel_max+holdVol_pre

        if tradeVol == 0:
            act = 0

        # 更新实际操作方向
        df.loc[df.index[k], 'act'] = act

        # 交易执行
        if act == 0:
            df.loc[df.index[k], 'holdVol'] = holdVol_pre
            df.loc[df.index[k], 'holdCost'] = holdCost_pre
            df.loc[df.index[k], 'holdCash'] = holdCash_pre
        elif act == -1:
            df = buy_act(df, k, buyPrice, tradeVol, holdVol_pre, holdCost_pre,
                         holdCash_pre)
        elif act == 1:
            df = sel_act(df, k, selPrice, tradeVol, holdVol_pre, holdCost_pre,
                         holdCash_pre)

        # 持仓信息更新
        holdVol = df.loc[df.index[k], 'holdVol']
        Price = df.loc[df.index[k], col_price]
        if holdVol > 0:
            df.loc[df.index[k], 'holdVal'] = holdVol * Price * (1-fee)
        elif holdVol < 0:
            df.loc[df.index[k], 'holdVal'] = holdVol * Price * (1+fee)

        df.loc[df.index[k], 'holdGainPct'] = cal_gain_pct(
           df.loc[df.index[k], 'holdCost'], df.loc[df.index[k], 'holdVal'], 0)

        # 盈亏和资金占用更新
        if k == 0:
            df.loc[df.index[k], 'cashGet_cum'] = df.loc[df.index[k], 'cashGet']
            df.loc[df.index[k], 'cashPut_cum'] = df.loc[df.index[k], 'cashPut']
            df.loc[df.index[k], 'cashUsedtmp'] = df.loc[df.index[k], 'cashNeed']
        else:
            df.loc[df.index[k], 'cashGet_cum'] = \
                                    df.loc[df.index[k-1], 'cashGet_cum'] + \
                                    df.loc[df.index[k], 'cashGet']
            df.loc[df.index[k], 'cashPut_cum'] = \
                                    df.loc[df.index[k-1], 'cashPut_cum'] + \
                                    df.loc[df.index[k], 'cashPut']
            df.loc[df.index[k], 'cashUsedtmp'] = \
                                    df.loc[df.index[k-1], 'cashUsedtmp'] + \
                                    df.loc[df.index[k], 'cashNeed']
        df.loc[df.index[k], 'gain_cum'] = \
                                    df.loc[df.index[k], 'cashGet_cum'] + \
                                    df.loc[df.index[k], 'holdVal'] - \
                                    df.loc[df.index[k], 'cashPut_cum']
        df.loc[df.index[k], 'tmpValue'] = \
                                    df.loc[df.index[k], 'gain_cum'] + \
                                    df.loc[df.index[k], 'cashUsedtmp']
        # 若存在空单本金亏完的情况，需要补资金
        if df.loc[df.index[k], 'tmpValue'] <= 0:
            df.loc[df.index[k], 'cashNeed'] = \
                                    df.loc[df.index[k], 'cashNeed'] - \
                                    df.loc[df.index[k], 'tmpValue'] + sos_money
            df.loc[df.index[k], 'holdCash'] = sos_money
        if k == 0:
            df.loc[df.index[k], 'cashUsedtmp'] = df.loc[df.index[k], 'cashNeed']
        else:
            df.loc[df.index[k], 'cashUsedtmp'] = \
                                    df.loc[df.index[k-1], 'cashUsedtmp'] + \
                                    df.loc[df.index[k], 'cashNeed']
        last_act = act
        last_acted = act if act != 0 else last_acted
        if act == 1:
            last_actPrice = selPrice
        elif act == -1:
            last_actPrice = buyPrice

    # 现有持仓平均成本（交易完成后）
    df['holdCost_mean'] = (df['holdCost'] / df['holdVol']).fillna(0)
    df['holdCost_mean'] = df[['holdCost_mean', 'holdVol']].apply(lambda x:
                            x['holdCost_mean'] if x['holdVol'] >= 0 else \
                            -x['holdCost_mean'], axis=1)

    # 减|平仓位盈亏百分比（用于计算百分比盈亏比和赢率）
    df['act_gain_pct'] = df[['act_gain_Cost', 'act_gain_Val']].apply(
      lambda x: cal_gain_pct(x['act_gain_Cost'], x['act_gain_Val'], 0), axis=1)

    df['cashUsedMax'] = df['cashNeed'].cumsum() # 实际最大资金占用
    df['pctGain_maxUsed'] = df[['gain_cum', 'cashUsedMax']].apply( lambda x:
      x_div_y(x['gain_cum'], x['cashUsedMax'], v_xy0=0), axis=1) # 收益/最大占用

    # 最大占用资金最大值
    cashUsedMax = df['cashUsedMax'].iloc[-1]

    # 账户总值（总投入+盈利）
    # 实际总值（用于计算基金净值）
    df['AccountValue_act'] = df['gain_cum'] + df['cashUsedMax']
    df['AccountValue_act_net'] = df['AccountValue_act'] / df['cashUsedMax']
    # 按最大成本算总值（用于根据账户总价值计算年化收益率、夏普、最大回撤等）
    df['AccountValue_maxUsed'] = df['gain_cum'] + cashUsedMax
    df['AccountValue_maxUsed_net'] = df['AccountValue_maxUsed'] / cashUsedMax

    # 持有仓位当期收益率（可用于累加和累乘收益计算）
    df['holdGainPct_cur'] = df['holdVal'].rolling(2).apply(lambda x:
                                    cal_gain_pct(x.iloc[0], x.iloc[1], pct_cost0=0))
    df.loc[df.index[0], 'holdGainPct_cur'] = df['holdGainPct'].iloc[0]
    for k in range(1, df.shape[0]):
        buyVol_cur = df['buyVol'].iloc[k]
        selVol_cur = df['selVol'].iloc[k]
        if buyVol_cur == 0 and selVol_cur == 0:
            continue
        else:
            preHoldVal = df['holdVal'].iloc[k-1] # 前期持仓总价值
            preHoldVol = df['holdVol'].iloc[k-1] # 前期持仓总量
            curHoldVal = df['holdVal'].iloc[k] # 当前持仓总价值
            tradeVol_cur = buyVol_cur - selVol_cur # 当期交易量
            if preHoldVol == 0: # 开仓
                df.loc[df.index[k], 'holdGainPct_cur'] = \
                                            df.loc[df.index[k], 'holdGainPct']
            elif preHoldVol > 0 and tradeVol_cur > 0: # buy加仓
                CostTotal = preHoldVal + df.loc[df.index[k], 'cashPut']
                df.loc[df.index[k], 'holdGainPct_cur'] = cal_gain_pct(
                                                CostTotal, curHoldVal, pct_cost0=0)
            elif preHoldVol < 0 and tradeVol_cur < 0: # sel加仓
                CostTotal = preHoldVal - df.loc[df.index[k], 'cashGet']
                df.loc[df.index[k], 'holdGainPct_cur'] = cal_gain_pct(
                                                CostTotal, curHoldVal, pct_cost0=0)
            elif preHoldVol > 0 and tradeVol_cur < 0 and \
                                    preHoldVol + tradeVol_cur >= 0: # buy减|平仓
                ValTotal = curHoldVal + df.loc[df.index[k], 'cashGet']
                df.loc[df.index[k], 'holdGainPct_cur'] = cal_gain_pct(
                                    preHoldVal, ValTotal, pct_cost0=0)
            elif preHoldVol < 0 and tradeVol_cur > 0 and \
                                    preHoldVol + tradeVol_cur <= 0: # sel减|平仓
                ValTotal = curHoldVal - df.loc[df.index[k], 'cashPut']
                df.loc[df.index[k], 'holdGainPct_cur'] = cal_gain_pct(
                                    preHoldVal, ValTotal, pct_cost0=0)
            elif preHoldVol > 0 and tradeVol_cur < 0 and \
                                    preHoldVol + tradeVol_cur < 0: # 平buy反向sel
                ValTotal = df.loc[df.index[k], 'cashGet']
                CostTotal = preHoldVal - curHoldVal
                df.loc[df.index[k], 'holdGainPct_cur'] = cal_gain_pct(
                                    CostTotal, ValTotal, pct_cost0=0)
            elif preHoldVol < 0 and tradeVol_cur > 0 and \
                                    preHoldVol + tradeVol_cur > 0: # 平sel反向buy
                CostTotal = df.loc[df.index[k], 'cashPut']
                ValTotal = curHoldVal - preHoldVal
                df.loc[df.index[k], 'holdGainPct_cur'] = cal_gain_pct(
                                    CostTotal, ValTotal, pct_cost0=0)

    totalGain = df['gain_cum'].iloc[-1] # 总收益额
    pctGain_maxUsed = df['pctGain_maxUsed'].iloc[-1]
    # 交易次数
    Nbuy = df[df['buyVol'] != 0].shape[0]
    Nsel = df[df['selVol'] != 0].shape[0]
    if not force_final0:
        if df['holdVol'].iloc[-1] > 0 and df['selVol'].iloc[-1] == 0:
            Nsel += 1
        elif df['holdVol'].iloc[-1] < 0 and df['buyVol'].iloc[-1] == 0:
            Nbuy += 1

    Nacts = Nbuy + Nsel
    # 平均赢率（总交易次数）

    Nliqs = df[df['act_gain_label'] != 0].shape[0]
    Nliqs_gain = df[df['act_gain_label'] == 1].shape[0]
    if not force_final0:
        if df['holdVol'].iloc[-1] > 0 and df['selVol'].iloc[-1] == 0:
            Nliqs += 1
            if df['holdVal'].iloc[-1] >= df['holdCost'].iloc[-1]:
                Nliqs_gain += 1
        elif df['holdVol'].iloc[-1] < 0 and df['buyVol'].iloc[-1] == 0:
            Nliqs += 1
            if df['holdVal'].iloc[-1] >= df['holdCost'].iloc[-1]:
                Nliqs_gain += 1

    # 盈亏比
    TotGain = df[df['act_gain'] > 0]['act_gain'].sum()
    TotLoss = df[df['act_gain'] < 0]['act_gain'].sum()
    if not force_final0:
        if df['holdVal'].iloc[-1] >= df['holdCost'].iloc[-1]:
            TotGain += (df['holdVal'].iloc[-1] - df['holdCost'].iloc[-1])
        else:
            TotLoss += (df['holdVal'].iloc[-1] - df['holdCost'].iloc[-1])
    TotLoss = abs(TotLoss)
    MeanGain = x_div_y(TotGain, Nliqs_gain, v_x0=0, v_y0=0, v_xy0=0)
    MeanLoss = x_div_y(TotLoss, Nliqs-Nliqs_gain, v_x0=0, v_y0=0, v_xy0=0)
    gain_loss_rate = x_div_y(MeanGain, MeanLoss, v_x0=0, v_y0=np.inf)

    # 百分比盈亏比
    TotGainPct = df[df['act_gain_pct'] > 0]['act_gain_pct'].sum()
    TotLossPct = df[df['act_gain_pct'] < 0]['act_gain_pct'].sum()
    if not force_final0:
        if df['holdVal'].iloc[-1] >= df['holdCost'].iloc[-1]:
            TotGainPct += cal_gain_pct(df['holdCost'].iloc[-1],
                                       df['holdVal'].iloc[-1], 0)
        else:
            TotLossPct += cal_gain_pct(df['holdCost'].iloc[-1],
                                       df['holdVal'].iloc[-1], 0)
    TotLossPct = abs(TotLossPct)
    MeanGainPct = x_div_y(TotGainPct, Nliqs_gain, v_x0=0, v_y0=0, v_xy0=0)
    MeanLossPct = x_div_y(TotLossPct, Nliqs-Nliqs_gain, v_x0=0, v_y0=0, v_xy0=0)
    gain_loss_rate_pct = x_div_y(MeanGainPct, MeanLossPct, v_x0=0, v_y0=np.inf)

    # 百分比平均赢率
    meanGainPct_pct = (TotGainPct - TotLossPct) / Nliqs

    # 计算开仓统计量时先对最后一个时间的强平价格做更正
    df_ = df.copy()
    if force_final0 == 'settle':
        if last_force == 'buy':
            df_.loc[df.index[-1], col_price_buy] = \
                                              df_.loc[df.index[-1], col_price]
        elif last_force == 'sel':
            df_.loc[df.index[-1], col_price_sel] = \
                                              df_.loc[df.index[-1], col_price]
    trade_gain_info_open = get_open_gain_info(df_, col_price=col_price,
            col_price_buy=col_price_buy, col_price_sel=col_price_sel, fee=fee,
            force_final0=force_final0, nshow=nshow, logger=logger)

    # 最大连续开仓次数
    df['act_n'] = df[['act', 'holdVol']].apply(lambda x:
                  np.nan if x['act'] != 0 and x['holdVol'] == 0 else \
                  (-1 if x['act'] == -1 and x['holdVol'] > 0 else \
                  (1 if x['act'] == 1 and x['holdVol'] < 0 else 0)), axis=1)
    df['con_buy_n'] = con_count_ignore(df['act_n'], lambda x: x == -1,
                                       func_ignore=lambda x: x == 0)
    df['con_sel_n'] = con_count_ignore(df['act_n'], lambda x: x == 1,
                                       func_ignore=lambda x: x == 0)

    trade_gain_info = {
            '收益/最大占用比': pctGain_maxUsed,
            '总收益': totalGain,
            '最大占用': cashUsedMax,
            'buy次数': Nbuy,
            'sel次数': Nsel,
            '总交易次数': Nacts,
            '平均赢率(总交易次数)': 2 * pctGain_maxUsed / Nacts,
            '百分比平均赢率(总交易次数)':
                ((TotGainPct-TotLossPct) + \
                 (trade_gain_info_open['盈利百分比和(开仓)']-\
                  trade_gain_info_open['亏损百分比和(开仓)'])) / \
                (Nliqs+trade_gain_info_open['开仓次数']),
            '平或减仓次数': Nliqs,
            '盈利次数(平仓或减仓)': Nliqs_gain,
            '亏损次数(平仓或减仓)': Nliqs - Nliqs_gain,
            '胜率(平仓或减仓)': Nliqs_gain / Nliqs,
            '平均赢率(平仓或减仓)': pctGain_maxUsed / Nliqs,
            '平均赢率(开仓)': pctGain_maxUsed / trade_gain_info_open['开仓次数'],
            '百分比平均赢率(平仓或减仓)': meanGainPct_pct,
            '盈亏比(平仓)': gain_loss_rate,
            '百分比盈亏比(平仓或减仓)': gain_loss_rate_pct,
            '总盈利额(平仓)': TotGain,
            '总亏损额(平仓)': TotLoss,
            '盈利百分比和(平仓或减仓)': TotGainPct,
            '亏损百分比和(平仓或减仓)': TotLossPct,
            '平均盈利(平仓)': MeanGain,
            '平均亏损(平仓)': MeanLoss,
            '平均盈利百分比(平仓或减仓)': MeanGainPct,
            '平均亏损百分比(平仓或减仓)': MeanLossPct,
            '单笔最大回撤(平仓或减仓)': df['holdPreMaxDown'].max(),
            '单笔最大亏损(平仓或减仓)': df['holdPreLossPctMax'].min(),
            '开仓频率': df.shape[0] / trade_gain_info_open['开仓次数'],
            '最大连续buy开(加)仓次数': df['con_buy_n'].max(),
            '最大连续sel开(加)次数': df['con_sel_n'].max()
            }
    trade_gain_info.update(trade_gain_info_open)

    df.index = ori_index

    return trade_gain_info, df

#%%
def get_open_gain_info(df_gain, col_price='close', col_price_buy='close',
                       col_price_sel='close', fee=1.5/1000,
                       force_final0='settle', nshow=None, logger=None):
    '''
    | 以开仓为统计口径，计算盈亏指标
    | ``df_gain`` 为 :func:`cal_sig_gains` 函数的输出 ``df``
    '''
    cols = list(set([col_price, col_price_buy, col_price_sel]))
    df = df_gain.reindex(columns=['buyVol', 'selVol', 'holdVol']+cols)
    holdVolLast = df['holdVol'].iloc[-1]
    if holdVolLast < 0:
        df.loc[df.index[-1], 'buyVol'] = df['buyVol'].iloc[-1] - holdVolLast
        df.loc[df.index[-1], 'holdVol'] = 0
        if force_final0 != 'trade':
            df.loc[df.index[-1], col_price_buy] = \
                                          df.loc[df.index[-1], col_price]
    elif holdVolLast > 0:
        df.loc[df.index[-1], 'selVol'] = df['selVol'].iloc[-1] + holdVolLast
        df.loc[df.index[-1], 'holdVol'] = 0
        if force_final0 != 'trade':
            df.loc[df.index[-1], col_price_sel] = \
                                          df.loc[df.index[-1], col_price]
    df['Tidx'] = range(0, df.shape[0])
    df['BuySelVol'] = df['buyVol'] - df['selVol']
    # df = df[df['BuySelVol'] != 0].copy()
    n_open, n_gain, n_loss = 0, 0, 0 # 记录开仓次数, 盈利次数, 亏损次数
    Vgain, Vloss, PctGain, PctLoss = 0, 0, 0, 0 # 记录总盈利和总亏损额及百分比之和
    i, N = 0, df.shape[0]
    hold_times_all = [] # 记录持仓周期
    maxdowns, maxlosss = [], [] # 记录分笔最大回撤和最大亏损
    while i < N:
        if nshow and i % nshow == 0:
            logger_show('GetOpenGainInfo: {} / {} ...'.format(i, df.shape[0]),
                        logger)

        volOpen = df['BuySelVol'].iloc[i] # 开仓量
        if volOpen == 0:
            i += 1
            continue

        n_open += 1
        # 开仓总成本
        if volOpen > 0:
            costOpen = df[col_price_buy].iloc[i] * volOpen * (1+fee)
        elif volOpen < 0:
            costOpen = df[col_price_sel].iloc[i] * volOpen * (1-fee)
        # 寻找对应平仓位置和量
        valLiqs = 0 # 平仓总值
        volLiq = 0
        maxdown, maxloss, maxgain = 0, 0, 0 # 最大回撤、最大亏损和最大盈利跟踪
        volLeft = abs(volOpen) - abs(volLiq)
        j = i+1
        hold_times = []
        while j < N and volLeft > 0:
            # 回撤更新
            Price_settle = df[col_price].iloc[j-1]
            if volOpen > 0:
                Value_settle = valLiqs - volLeft * Price_settle * (1-fee)
            else:
                Value_settle = valLiqs + volLeft * Price_settle * (1+fee)
            pct = cal_gain_pct(costOpen, -Value_settle, pct_cost0=0)
            maxloss = min(pct, maxloss)
            maxgain = max(pct, maxgain)
            if volOpen > 0:
                valMax = costOpen * (1 + maxgain)
            elif volOpen < 0:
                valMax = costOpen * (1 - maxgain)
            maxdown = abs(cal_gain_pct(valMax, -Value_settle, pct_cost0=0))

            if df['BuySelVol'].iloc[j] == 0:
                j += 1
                continue

            # 操作方向相反则纳入平仓量
            vol_ = df['BuySelVol'].iloc[j]
            if volOpen * vol_ < 0:
                volLiq += df['BuySelVol'].iloc[j]
                volLeft = abs(volOpen) - abs(volLiq)
                if volLeft == 0: # 刚好完成平仓
                    df.loc[df.index[j], 'BuySelVol'] = 0
                    if volOpen > 0:
                        valLiqs += df[col_price_sel].iloc[j] * vol_ * (1-fee)
                    elif volOpen < 0:
                        valLiqs += df[col_price_buy].iloc[j] * vol_ * (1+fee)
                    if costOpen + valLiqs > 0:
                        n_loss += 1
                        Vloss += (costOpen + valLiqs)
                        pct = cal_gain_pct(costOpen, -valLiqs, pct_cost0=0)
                        PctLoss += abs(pct)
                        maxloss = min(pct, maxloss)
                        maxgain = max(pct, maxgain)
                        if volOpen > 0:
                            valMax = costOpen * (1 + maxgain)
                        elif volOpen < 0:
                            valMax = costOpen * (1 - maxgain)
                        maxdown = abs(cal_gain_pct(valMax, -valLiqs, pct_cost0=0))
                    else:
                        n_gain += 1
                        Vgain += abs(costOpen + valLiqs)
                        PctGain += cal_gain_pct(costOpen, -valLiqs, pct_cost0=0)
                    hold_times.append(df.loc[df.index[j], 'Tidx']- \
                                  df.loc[df.index[i], 'Tidx'])
                elif volLeft > 0: # 未完成平仓
                    df.loc[df.index[j], 'BuySelVol'] = 0
                    if volOpen > 0:
                        valLiqs += df[col_price_sel].iloc[j] * vol_ * (1-fee)
                    elif volOpen < 0:
                        valLiqs += df[col_price_buy].iloc[j] * vol_ * (1+fee)
                    hold_times.append(df.loc[df.index[j], 'Tidx']- \
                                  df.loc[df.index[i], 'Tidx'])
                    j += 1
                elif volLeft < 0: # 完成平仓且开新仓
                    if volOpen > 0:
                        df.loc[df.index[j], 'BuySelVol'] = volLeft
                        vol__ = vol_ - volLeft
                        valLiqs += df[col_price_sel].iloc[j] * vol__ * (1-fee)
                    elif volOpen < 0:
                        df.loc[df.index[j], 'BuySelVol'] = -volLeft
                        vol__ = vol_ + volLeft
                        valLiqs += df[col_price_buy].iloc[j] * vol__ * (1+fee)
                    if costOpen + valLiqs > 0:
                        n_loss += 1
                        Vloss += (costOpen + valLiqs)
                        pct = cal_gain_pct(costOpen, -valLiqs, pct_cost0=0)
                        PctLoss += abs(pct)
                        maxloss = min(pct, maxloss)
                        maxgain = max(pct, maxgain)
                        if volOpen > 0:
                            valMax = costOpen * (1 + maxgain)
                        elif volOpen < 0:
                            valMax = costOpen * (1 - maxgain)
                        maxdown = abs(cal_gain_pct(valMax, -valLiqs, pct_cost0=0))
                    else:
                        n_gain += 1
                        Vgain += abs(costOpen + valLiqs)
                        PctGain += cal_gain_pct(costOpen, -valLiqs, pct_cost0=0)
                    hold_times.append(df.loc[df.index[j], 'Tidx']- \
                                  df.loc[df.index[i], 'Tidx'])
            else:
                j += 1
        hold_times_all.append(hold_times)
        maxdowns.append(maxdown)
        maxlosss.append(maxloss)
        i += 1
    Mgain = x_div_y(Vgain, n_gain, v_x0=0, v_y0=0, v_xy0=0)
    Mloss = x_div_y(Vloss, n_loss, v_x0=0, v_y0=0, v_xy0=0)
    Mgain_pct = x_div_y(PctGain, n_gain, v_x0=0, v_y0=0, v_xy0=0)
    Mloss_pct = x_div_y(PctLoss, n_loss, v_x0=0, v_y0=0, v_xy0=0)
    mean_hold_time1 = sum([sum(x) for x in hold_times_all]) / \
                                    sum([len(x) for x in hold_times_all])
    mean_hold_time2 = np.mean([np.mean(x) for x in hold_times_all])
    gain_info = {'开仓次数': n_open,
                 '盈利次数(开仓)': n_gain,
                 '亏损次数(开仓)': n_loss,
                 '胜率(开仓)': n_gain / n_open,
                 '总盈利额(开仓)': Vgain,
                 '总亏损额(开仓)': Vloss,
                 '盈利百分比和(开仓)': PctGain,
                 '亏损百分比和(开仓)': PctLoss,
                 '平均盈利(开仓)': Mgain,
                 '平均亏损(开仓)': Mloss,
                 '平均盈利百分比(开仓)': Mgain_pct,
                 '平均亏损百分比(开仓)': Mloss_pct,
                 '盈亏比(开仓)': x_div_y(Mgain, Mloss, v_x0=0, v_y0=np.inf),
                 '百分比盈亏比(开仓)': x_div_y(Mgain_pct, Mloss_pct,
                                                  v_x0=0, v_y0=np.inf),
                 '百分比平均赢率(开仓)': (PctGain-PctLoss) / n_open,
                 '平均持仓周期(所有平均)': mean_hold_time1,
                 '平均持仓周期(每次平均)': mean_hold_time2,
                 '单笔最大回撤(开仓)': max(maxdowns),
                 '单笔最大亏损(开仓)': min(maxlosss)}

    return gain_info

#%%
def get_yield_curve(data, sig_col, nn=252, ext_type=1,
                    net_type='fundnet', gain_type='pct', rtype='exp',
                    show_sigs=True, show_dy_maxdown=False, show_key_infos=True,
                    logger=None, plot=True, kwargs_plot={'figsize': (11, 7)},
                    **kwargs_gain):
    '''
    根据信号生成收益曲线
    '''

    assert net_type in ['fundnet', 'prod', 'sum']

    trade_gain_info, df_gain = cal_sig_gains(data, sig_col, **kwargs_gain)

    cols_to_plot = []
    for col_key in ['cols_styl_up_left', 'cols_styl_up_right',
                    'cols_styl_low_left', 'cols_styl_low_right',
                    'cols_to_label_info', 'xparls_info']:
        if col_key in kwargs_plot.keys():
            if col_key == 'cols_to_label_info':
                for col_name in kwargs_plot[col_key].keys():
                    cols_to_plot.append(col_name)
                    for lbl_info in kwargs_plot[col_key][col_name]:
                        cols_to_plot.append(kwargs_plot[col_key][col_name][0][0])
            else:
                for col_name in kwargs_plot[col_key]:
                    cols_to_plot.append(col_name)
    cols_to_plot = list(set(cols_to_plot))
    for col_name in cols_to_plot:
        if col_name in df_gain.columns:
            logger_show('{}列画图数据被更新！'.format(col_name), logger, 'warn')
            continue
        df_gain[col_name] = data[col_name]

    if 'col_price' in kwargs_gain.keys():
        col_price = kwargs_gain['col_price']
    else:
        col_price = 'close'

    if net_type == 'sum':
        df_gain['value_mkt'] = \
            1 + df_gain[col_price].pct_change().cumsum().fillna(0)
        df_gain['AccountValue_maxUsed_net1'] = \
            1 + df_gain['AccountValue_maxUsed'].pct_change().cumsum().fillna(0)
    else:
        df_gain['value_mkt'] = df_gain[col_price] / df_gain[col_price].iloc[0]
        df_gain['AccountValue_maxUsed_net1'] = df_gain['AccountValue_maxUsed_net'].copy()

    df_gain['转入'] = df_gain['cashNeed']
    df_gain['转出'] = 0
    df_gain['资产总值'] = df_gain['AccountValue_act']
    df_gain['盈亏%'] = df_gain['holdGainPct_cur'] * 100
    df_gain = get_gains(df_gain, gain_types=['fundnet', 'prod', 'sum', 'act'])
    df_gain['基金净值'] = df_gain['净值'].copy()

    if net_type == 'prod':
        df_gain['净值'] = df_gain['累乘净值'].copy()
    elif net_type == 'sum':
        df_gain['净值'] = df_gain['累加净值'].copy()
        gain_type = 'pct'
        rtype = 'mean'
    MDabsV = True if net_type == 'sum' else False

    if net_type in ['sum', 'prod']:
        # 按百分比算每期盈亏比和每期平均赢率
        TotGain = df_gain[df_gain['holdGainPct_cur'] > 0]['holdGainPct_cur'].sum()
        TotLoss = abs(df_gain[df_gain['holdGainPct_cur'] < 0]['holdGainPct_cur'].sum())
        Nhit = df_gain[df_gain['holdGainPct_cur'] > 0].shape[0]
        Nlos = df_gain[df_gain['holdGainPct_cur'] < 0].shape[0]
        MeanGain = x_div_y(TotGain, Nhit, v_x0=0, v_y0=0, v_xy0=0)
        MeanLoss = x_div_y(TotLoss, Nlos, v_x0=0, v_y0=0, v_xy0=0)
        gain_loss_rate_pct = x_div_y(MeanGain, MeanLoss, v_x0=0, v_y0=np.inf)

        trade_gain_info.update(
            {'盈亏比(百分比每期)': gain_loss_rate_pct,
             '平均赢率(百分比每期)': (TotGain-TotLoss) / (Nhit+Nlos)})

    # 年化收益
    return_ann_maxUsed = cal_returns_period(df_gain['AccountValue_maxUsed_net1'],
                                  gain_type=gain_type, rtype=rtype, nn=nn)
    return_ann_net = cal_returns_period(df_gain['净值'], gain_type=gain_type,
                                        rtype=rtype, nn=nn)
    return_ann_mkt = cal_returns_period(df_gain['value_mkt'], gain_type=gain_type,
                                        rtype=rtype, nn=nn)
    # 夏普
    sharpe_maxUsed = cal_sharpe(df_gain['AccountValue_maxUsed_net1'], r=3/100,
                                nn=nn, gain_type=gain_type, ann_rtype=rtype)
    sharpe_net = cal_sharpe(df_gain['净值'], r=3/100, nn=nn,
                            gain_type=gain_type, ann_rtype=rtype)
    sharpe_mkt = cal_sharpe(df_gain['value_mkt'], r=3/100, nn=nn,
                            gain_type=gain_type, ann_rtype=rtype)
    # 最大回撤
    maxDown_maxUsed, (strt_idx_maxUsed, end_idx_maxUsed) = get_maxdown(
                            df_gain['AccountValue_maxUsed_net1'], abs_val=MDabsV)
    maxDown_net, (strt_idx_net, end_idx_net) = get_maxdown(df_gain['净值'],
                                                           abs_val=MDabsV)
    maxDown_mkt, (strt_idx_mkt, end_idx_mkt) = get_maxdown(df_gain['value_mkt'],
                                                           abs_val=MDabsV)

    ori_index = df_gain.index
    df_gain.index = range(0, df_gain.shape[0])

    # 回撤标注
    df_gain['inMaxDown_net'] = 0
    df_gain.loc[df_gain.index[strt_idx_net: end_idx_net+1], 'inMaxDown_net'] = 1
    df_gain['inMaxDown_mkt'] = 0
    df_gain.loc[df_gain.index[strt_idx_mkt: end_idx_mkt+1], 'inMaxDown_mkt'] = 1

    # 动态最大回撤
    df_gain['dyMaxDown'] = get_maxdown_dy(df_gain['净值'])

    df_gain.index = ori_index

    if plot:
        plot_series(df_gain,
                    {'AccountValue_act_net': ('-m', '账户净值(价值/实际最大占用)'),
                     'AccountValue_maxUsed_net': ('-r', '账户净值(价值/最终最大占用)'),
                     '基金净值': ('-b', '账户净值(基金份额法)'),
                     '累加净值': ('-c', '账户净值(累加净值)'),
                     '累乘净值': ('-y', '账户净值(累乘净值)'),
                     'value_mkt': ('-k', '市场净值(价格/初始价格)')},
                    **kwargs_plot)

        if show_sigs:
            sigs_label_info = {
                    'value_mkt':
                    [['act', (-1, 1), ('r^', 'gv'), ('买', '卖')],
                      ['act_stop', (-1.5, 1.5, -0.5, 0.5), ('r*', 'g*', 'mo', 'bo'),
                      ('做空止盈', '做多止盈', '做空止损', '做多止损')]]}
        else:
            sigs_label_info = {}
        conlabel_info = {
            '净值':
                [['inMaxDown_net', (1, 0), ('-m', '-b'), ('最大回撤区间', '账户净值')]],
            'value_mkt':
                [['inMaxDown_mkt', (1, 0), ('-m', '-k'), (False, 'market')]]
                }
        conlabel_info, kwargs_plot = get_update_kwargs('conlabel_info',
                                                       conlabel_info, kwargs_plot)
        cols_to_label_info, kwargs_plot = get_update_kwargs('cols_to_label_info',
                                                    sigs_label_info, kwargs_plot)
        cols_styl_up_left = {'value_mkt': ('-w', False),
                             '净值': ('-w', False)}
        cols_styl_up_left, kwargs_plot = get_update_kwargs('cols_styl_up_left',
                                              cols_styl_up_left, kwargs_plot)
        if show_dy_maxdown:
            cols_styl_up_right = {'dyMaxDown': ('-c', '动态最大回撤',
                                                {'alpha': 0.2})}
            cols_to_fill_info = {'dyMaxDown': {'color': 'c', 'alpha': 0.2}}
        else:
            cols_styl_up_right = {}
            cols_to_fill_info = {}
        cols_styl_up_right, kwargs_plot = get_update_kwargs('cols_styl_up_right',
                                              cols_styl_up_right, kwargs_plot)
        cols_to_fill_info, kwargs_plot = get_update_kwargs('cols_to_fill_info',
                                              cols_to_fill_info, kwargs_plot)

        plot_series_conlabel(df_gain, conlabel_info=conlabel_info,
                             cols_styl_up_left=cols_styl_up_left,
                             cols_styl_up_right=cols_styl_up_right,
                             cols_to_label_info=cols_to_label_info,
                             cols_to_fill_info=cols_to_fill_info,
                             **kwargs_plot)

    gain_stats = pd.DataFrame({
        '账户(最大占用)': [return_ann_maxUsed, sharpe_maxUsed, maxDown_maxUsed],
        '账户(净值)': [return_ann_net, sharpe_net, maxDown_net],
        '市场': [return_ann_mkt, sharpe_mkt, maxDown_mkt]})
    gain_stats.index = ['年化收益', '夏普', '最大回撤']

    # 超额收益
    extr = cal_ext_return_period(df_gain['净值'], df_gain['value_mkt'],
                                 gain_type=gain_type, rtype=rtype,
                                 nn=nn, ext_type=ext_type)
    trade_gain_info.update({'年化超额': extr})

    trade_gain_info.update({'最大回撤区间长度': df_gain['inMaxDown_net'].sum()})

    # alpha，beta
    alpha, beta = cal_alpha_beta(df_gain['净值'], df_gain['value_mkt'],
                                 gain_type=gain_type, rtype=rtype,
                                 nn=nn, logger=logger)

    trade_gain_info.update({
        '年化收益(净值)': gain_stats.loc['年化收益', '账户(净值)'],
        '年化收益(最大占用)': gain_stats.loc['年化收益', '账户(最大占用)'],
        '年化收益(市场)': gain_stats.loc['年化收益', '市场'],
        '夏普(净值)': gain_stats.loc['夏普', '账户(净值)'],
        '夏普(最大占用)': gain_stats.loc['夏普', '账户(最大占用)'],
        '夏普(市场)': gain_stats.loc['夏普', '市场'],
        '最大回撤(净值)': gain_stats.loc['最大回撤', '账户(净值)'],
        '最大回撤(最大占用)': gain_stats.loc['最大回撤', '账户(最大占用)'],
        '最大回撤(市场)': gain_stats.loc['最大回撤', '市场'],
        'alpha': alpha,
        'beta': beta
        })

    if show_key_infos:
        _print_key_infos(trade_gain_info, net_type=net_type)

    return trade_gain_info, df_gain

#%%
def _print_key_infos(trade_gain_info, net_type='fundnet'):
    print('年化收益率：{}；'.format(round(trade_gain_info['年化收益(净值)'], 4)) + \
          '年化收益率（市场）：{}'.format(round(trade_gain_info['年化收益(市场)'], 4)))
    print('年化收益率（超额）：{}'.format(round(trade_gain_info['年化超额'], 4)))
    print('最大回撤：{}；'.format(round(trade_gain_info['最大回撤(净值)'], 4)) + \
          '最大回撤（市场）：{}'.format(round(trade_gain_info['最大回撤(市场)'], 4)))
    print('胜率(平)：{}；'.format(round(trade_gain_info['胜率(平仓或减仓)'], 4)) + \
          '胜率(开)：{}'.format(round(trade_gain_info['胜率(开仓)'], 4)))
    if net_type in ['prod', 'sum']:
        print('盈亏比(平): {}；'.format(round(trade_gain_info['百分比盈亏比(平仓或减仓)'], 4)) + \
              '平均赢率：{}'.format(round(trade_gain_info['百分比平均赢率(总交易次数)'], 4)))
    else:
        print('盈亏比(平): {}；'.format(round(trade_gain_info['盈亏比(平仓)'], 4)) + \
              '平均赢率：{}'.format(round(trade_gain_info['平均赢率(总交易次数)'], 4)))
    print('夏普比率：{}；'.format(round(trade_gain_info['夏普(净值)'], 4)) + \
          '夏普比率（市场）：{}'.format(round(trade_gain_info['夏普(市场)'], 4)))
    print('单笔最大回撤(平): {}；'.format(round(trade_gain_info['单笔最大回撤(平仓或减仓)'], 4)) + \
          '单笔最大亏损(平)：{}'.format(round(trade_gain_info['单笔最大亏损(平仓或减仓)'], 4)))
    print('最大回撤区间长度: {}'.format(round(trade_gain_info['最大回撤区间长度'], 4)))
    print('平均持仓周期: {}；'.format(round(trade_gain_info['平均持仓周期(所有平均)'], 4)) + \
          '开仓频率: {}'.format(round(trade_gain_info['开仓频率'], 4)))

#%%
def get_yield_curve2(data, col_gain, col_cost, col_price=None, nn=252,
                     net_type='fundnet', gain_type='pct', rtype='exp',
                     ext_type=1, show_mkt=False, logger=None,
                     show_dy_maxdown=False, show_key_infos=True, plot=True,
                     kwargs_plot={}):
    '''根据每期收益和成本/资金占用计算收益曲线'''

    assert net_type in ['fundnet', 'prod', 'sum']

    # 价格列检查
    if isnull(col_price) and \
                (show_mkt or (ext_type is not False and not isnull(ext_type))):
        logger_show('未识别价格列，收益曲线无法与市场基准比较！', logger, 'warn')
    if isnull(col_price):
        show_mkt = False
        df = data.reindex(columns=[col_gain, col_cost])
    else:
        df = data.reindex(columns=[col_gain, col_cost, col_price])

    df[col_gain] = df[col_gain].fillna(0)
    df[col_cost] = df[col_cost].fillna(0)

    ori_idx = df.index
    df.index = range(0, df.shape[0])

    # 净值及收益计算
    df['转入'] = 0
    df.loc[df.index[0], '转入'] = df[col_cost].iloc[0]
    df['资产总值'] = 0
    df.loc[df.index[0], '资产总值'] = df['转入'].iloc[0] + df[col_gain].iloc[0]
    for k in range(1, df.shape[0]):
        if df['资产总值'].iloc[k-1] >= df[col_cost].iloc[k]:
            df.loc[df.index[k], '转入'] = 0
        else:
            df.loc[df.index[k], '转入'] = df[col_cost].iloc[k] - \
                                                        df['资产总值'].iloc[k-1]
        df.loc[df.index[k], '资产总值'] = df['转入'].iloc[k] + \
                                 df['资产总值'].iloc[k-1] + df[col_gain].iloc[k]
    df['转出'] = 0
    df['GainPct'] = df[[col_gain, col_cost]].apply(lambda x:
                    x_div_y(x[col_gain], x[col_cost], v_x0=0, v_y0=0, v_xy0=0),
                    axis=1)
    df['盈亏%'] = df['GainPct'] * 100
    df = get_gains(df, gain_types=['fundnet', 'prod', 'sum', 'act'])
    df['基金净值'] = df['净值'].copy()
    if net_type == 'prod':
        df['净值'] = df['累乘净值'].copy()
    elif net_type == 'sum':
        df['净值'] = df['累加净值']
        gain_type = 'pct'
        rtype = 'mean'
    MDabsV = True if net_type == 'sum' else False

    df['cashUsedMax'] = df['转入'].cumsum() # 实际最大资金占用
    df['AccountValue_act_net'] = df['资产总值'] / df['cashUsedMax']
    cashUsedMax = df['cashUsedMax'].iloc[-1]
    df['AccountValue_maxUsed'] = df[col_gain].cumsum() + cashUsedMax
    df['AccountValue_maxUsed_net'] = df['AccountValue_maxUsed'] / cashUsedMax

    if net_type == 'sum':
            df['AccountValue_maxUsed_net1'] = \
                1 + df['AccountValue_maxUsed'].pct_change().cumsum().fillna(0)
    else:
        df['AccountValue_maxUsed_net1'] = df['AccountValue_maxUsed_net'].copy()

    # 年化收益
    return_ann = cal_returns_period(df['净值'], nn=nn,
                                    gain_type=gain_type, rtype=rtype)
    return_ann_maxUsed = cal_returns_period(df['AccountValue_maxUsed_net1'],
                                  nn=nn, gain_type=gain_type, rtype=rtype)
    # 夏普
    sharpe = cal_sharpe(df['净值'], r=3/100, nn=nn,
                        gain_type=gain_type, ann_rtype=rtype)
    sharpe_maxUsed = cal_sharpe(df['AccountValue_maxUsed_net1'], r=3/100,
                                nn=nn, gain_type=gain_type, ann_rtype=rtype)
    # 最大回撤
    maxDown, (strt_idx, end_idx) = get_maxdown(df['净值'], abs_val=MDabsV)
    df['inMaxDown'] = 0
    df.loc[df.index[strt_idx: end_idx+1], 'inMaxDown'] = 1
    maxDown_maxUsed, (strt_idx_maxUsed, end_idx_maxUsed) = get_maxdown(
                                  df['AccountValue_maxUsed_net1'], abs_val=MDabsV)

    # 动态最大回撤
    df['dyMaxDown'] = get_maxdown_dy(df['净值'])

    if not isnull(col_price):
        df['value_mkt'] = df[col_price] / df[col_price].iloc[0]
        if net_type == 'sum':
            df['value_mkt'] = \
                            1 + df[col_price].pct_change().cumsum().fillna(0)
        return_ann_mkt = cal_returns_period(df['value_mkt'], nn=nn,
                                            gain_type=gain_type, rtype=rtype)
        sharpe_mkt = cal_sharpe(df['value_mkt'], r=3/100, nn=nn,
                                gain_type=gain_type, ann_rtype=rtype)
        maxDown_mkt, (strt_idx_mkt, end_idx_mkt) = get_maxdown(df['value_mkt'],
                                                               abs_val=MDabsV)
        df['inMaxDown_mkt'] = 0
        df.loc[df.index[strt_idx_mkt: end_idx_mkt+1], 'inMaxDown_mkt'] = 1
        extr = cal_ext_return_period(df['净值'], df['value_mkt'],
                                     gain_type=gain_type, rtype=rtype,
                                     nn=nn, ext_type=ext_type)
    else:
        return_ann_mkt = np.nan
        sharpe_mkt = np.nan
        maxDown_mkt = np.nan
        extr = np.nan

    Nhit = df[(df[col_cost] != 0) & (df[col_gain] >= 0)].shape[0]
    Nlos = df[(df[col_cost] != 0) & (df[col_gain] < 0)].shape[0]
    Ntot = Nhit + Nlos
    hit_rate = Nhit / Ntot
    sumGain = df[df[col_gain] > 0][col_gain].sum()
    sumLoss = abs(df[df[col_gain] < 0][col_gain].sum())

    if net_type in ['prod', 'sum']:
        TotGain = df[df[col_gain] > 0]['GainPct'].sum()
        TotLoss = abs(df[df[col_gain] < 0]['GainPct'].sum())
    else:
        TotGain = sumGain
        TotLoss = sumLoss
    MeanGain = x_div_y(TotGain, Nhit, v_x0=0, v_y0=0, v_xy0=0)
    MeanLoss = x_div_y(TotLoss, Nlos, v_x0=0, v_y0=0, v_xy0=0)
    gain_loss_rate = x_div_y(MeanGain, MeanLoss, v_x0=0, v_y0=np.inf)

    maxUsed = df['转入'].sum()
    finalGain = df[col_gain].sum()

    if net_type in ['prod', 'sum']:
        meanGainPct = df['GainPct'].sum() / Ntot
    else:
        meanGainPct = df[col_gain].sum() / df['转入'].sum() / Ntot

    gain_info = {'年化收益': return_ann,
                 '夏普': sharpe,
                 '最大回撤': maxDown,
                 '年化收益(市场)': return_ann_mkt,
                 '夏普(市场)': sharpe_mkt,
                 '最大回撤(市场)': maxDown_mkt,
                 '年化收益(最大占用)': return_ann_maxUsed,
                 '夏普(最大占用)': sharpe_maxUsed,
                 '最大回撤(最大占用)': maxDown_maxUsed,
                 '年化超额': extr,
                 '胜率': hit_rate,
                 '盈亏比': gain_loss_rate,
                 '平均赢率': meanGainPct,
                 '最大回撤区间长度': df['inMaxDown'].sum(),
                 '收益/最大占用比': finalGain / maxUsed,
                 '总收益': finalGain,
                 '最大占用': maxUsed,
                 '总(交易)期数': Ntot,
                 '盈利期数': Nhit,
                 '亏损期数': Nlos,
                 '总盈利额': sumGain,
                 '总亏损额': sumLoss,
                 '平均赢率额': x_div_y(sumGain, Nhit, v_x0=0, v_y0=0, v_xy0=0),
                 '平均亏损额': x_div_y(sumLoss, Nlos, v_x0=0, v_y0=0, v_xy0=0),
                 '单期最大亏损': df['GainPct'].min()}

    df.index = ori_idx

    if plot:
        cols_styl_up_left = \
            {'AccountValue_act_net': ('-m', '账户净值(价值/实际最大占用)'),
             'AccountValue_maxUsed_net': ('-r', '账户净值(价值/最终最大占用)'),
             '基金净值': ('-b', '账户净值(基金份额法)'),
             '累加净值': ('-c', '账户净值(累加净值)'),
             '累乘净值': ('-y', '账户净值(累乘净值)')}
        if not isnull(col_price):
            cols_styl_up_left.update(
                    {'value_mkt': ('-k', '市场净值(价格/初始价格)')})
        plot_series(df, cols_styl_up_left, **kwargs_plot)

        cols_to_plot = []
        for col_key in ['cols_styl_up_left', 'cols_styl_up_right',
                        'cols_styl_low_left', 'cols_styl_low_right',
                        'cols_to_label_info', 'xparls_info']:
            if col_key in kwargs_plot.keys():
                if col_key == 'cols_to_label_info':
                    for col_name in kwargs_plot[col_key].keys():
                        cols_to_plot.append(col_name)
                        for lbl_info in kwargs_plot[col_key][col_name]:
                            cols_to_plot.append(kwargs_plot[col_key][col_name][0][0])
                else:
                    for col_name in kwargs_plot[col_key]:
                        cols_to_plot.append(col_name)
        cols_to_plot = list(set(cols_to_plot))
        for col_name in cols_to_plot:
            if col_name in df.columns:
                logger_show('{}列画图数据被更新！'.format(col_name), logger, 'warn')
                continue
            df[col_name] = data[col_name]

        conlabel_info = {'净值':
             [['inMaxDown', (1, 0), ('-m', '-b'), ('最大回撤区间', '账户净值')]]}
        if not isnull(col_price) and show_mkt:
            conlabel_info.update({'value_mkt':
                [['inMaxDown_mkt', (1, 0), ('-m', '-k'), (False, 'market')]]})
        conlabel_info, kwargs_plot = get_update_kwargs(
                                     'conlabel_info', conlabel_info, kwargs_plot)
        cols_styl_up_left = {'净值': ('-w', False)}
        if not isnull(col_price) and show_mkt:
            cols_styl_up_left.update({'value_mkt': ('-w', False)})
        cols_styl_up_left, kwargs_plot = get_update_kwargs('cols_styl_up_left',
                                              cols_styl_up_left, kwargs_plot)
        cols_styl_up_right = {}
        cols_to_fill_info = {}
        if show_dy_maxdown:
            cols_styl_up_right.update({'dyMaxDown': ('-c', '动态最大回撤',
                                                     {'alpha': 0.2})})
            cols_to_fill_info.update({'dyMaxDown': {'color': 'c', 'alpha': 0.2}})
        cols_styl_up_right, kwargs_plot = get_update_kwargs('cols_styl_up_right',
                                              cols_styl_up_right, kwargs_plot)
        cols_to_fill_info, kwargs_plot = get_update_kwargs('cols_to_fill_info',
                                              cols_to_fill_info, kwargs_plot)

        plot_series_conlabel(df, conlabel_info=conlabel_info,
                             cols_styl_up_left=cols_styl_up_left,
                             cols_styl_up_right=cols_styl_up_right,
                             cols_to_fill_info=cols_to_fill_info,
                             **kwargs_plot)

    if show_key_infos:
        if not isnull(col_price) and show_mkt:
            print('年化收益率：{}；'.format(round(gain_info['年化收益'], 4)) + \
                  '年化收益率（市场）：{}'.format(round(gain_info['年化收益(市场)'], 4)))
            print('年化收益率（超额）：{}'.format(round(gain_info['年化超额'], 4)))
            print('最大回撤：{}；'.format(round(gain_info['最大回撤'], 4)) + \
                  '最大回撤（市场）：{}'.format(round(gain_info['最大回撤(市场)'], 4)))
            print('胜率：{}；'.format(round(gain_info['胜率'], 4)) + \
                  '盈亏比: {}'.format(round(gain_info['盈亏比'], 4)))
            print('平均赢率：{}'.format(round(gain_info['平均赢率'], 4)))
            print('夏普比率：{}；'.format(round(gain_info['夏普'], 4)) + \
                  '夏普比率（市场）：{}'.format(round(gain_info['夏普(市场)'], 4)))
            print('最大回撤区间长度: {}'.format(round(gain_info['最大回撤区间长度'], 4)))
        else:
            print('年化收益率：{}'.format(round(gain_info['年化收益'], 4)))
            print('最大回撤：{}'.format(round(gain_info['最大回撤'], 4)))
            print('胜率：{}；'.format(round(gain_info['胜率'], 4)) + \
                  '盈亏比: {}'.format(round(gain_info['盈亏比'], 4)))
            print('平均赢率：{}'.format(round(gain_info['平均赢率'], 4)))
            print('夏普比率：{}'.format(round(gain_info['夏普'], 4)))
            print('最大回撤区间长度: {}'.format(round(gain_info['最大回撤区间长度'], 4)))

    return gain_info, df

#%%
if __name__ == '__main__':
    import time
    from pprint import pprint
    from dramkit import load_csv

    strt_tm = time.time()

    #%%
    # 最大回撤测试
    values = [1.0, 1.01, 1.05, 1.1, 1.11, 1.07, 1.03, 1.03, 1.01, 1.02, 1.04,
              1.05, 1.07, 1.06, 1.05, 1.06, 1.07, 1.09, 1.12, 1.18, 1.15,
              1.15, 1.18, 1.16, 1.19, 1.17, 1.17, 1.18, 1.19, 1.23, 1.24,
              1.25, 1.24, 1.25, 1.24, 1.25, 1.24, 1.25, 1.24, 1.27, 1.23,
              1.22, 1.18, 1.2, 1.22, 1.25, 1.25, 1.27, 1.26, 1.31, 1.32, 1.31,
              1.33, 1.33, 1.36, 1.33, 1.35, 1.38, 1.4, 1.42, 1.45, 1.43, 1.46,
              1.48, 1.52, 1.53, 1.52, 1.55, 1.54, 1.53, 1.55, 1.54, 1.52,
              1.53, 1.53, 1.5, 1.45, 1.43, 1.42, 1.41, 1.43, 1.42, 1.45, 1.45,
              1.49, 1.49, 1.51, 1.54, 1.53, 1.56, 1.52, 1.53, 1.58, 1.58,
              1.58, 1.61, 1.63, 1.61, 1.59]
    data = pd.DataFrame(values, columns=['values'])
    # maxDown, (strt_idx, end_idx) = get_maxdown(data['values'])
    maxDown, (strt_idx, end_idx) = get_maxup(data['values'])
    data['in_maxDown'] = 0
    data.loc[data.index[strt_idx: end_idx+1], 'in_maxDown'] = 1
    plot_series(data, {'values': '.-b'},
                cols_to_label_info={'values':
                            [['in_maxDown', (1,), ('.-r',), ('最大回撤区间',)]]},
                grids=True, figsize=(11, 7))

    fpath = '../test/510050_daily_pre_fq.csv'
    data = load_csv(fpath)
    data.set_index('date', drop=False, inplace=True)
    data = data.iloc[-500:, :][['close']]

    plot_series(data, {'close': '.-b'}, grids=True, figsize=(11, 7))

    maxDown, _ , (strt_idx, end_idx) = get_maxdown_pd(data['close'])
    maxDown = str(round(maxDown, 4))
    data['in_maxDown'] = 0
    data.loc[data.index[strt_idx: end_idx+1], 'in_maxDown'] = 1
    data['maxdown'] = get_maxdown_all(data['close'])
    data['dyMaxDown'] = get_maxdown_dy(data['close'])
    plot_series(data, {'close': '.-b'},
                cols_styl_up_right={'maxdown': '.-y', 'dyMaxDown': '.-m'},
                cols_to_label_info={'close':
                            [['in_maxDown', (1,), ('.-r',), ('最大回撤区间',)]]},
                title='最大回撤：' + ' --> '.join(_) + ' (' + maxDown + ')',
                grids=True, figsize=(11, 7))

    # maxDown, (strt_idx, end_idx) = get_maxdown(data['close'])
    maxDown, (strt_idx, end_idx) = get_maxup(data['close'], abs_val=True)
    maxDown = str(round(maxDown, 4))
    _ = (data.index[strt_idx], data.index[end_idx])
    data['in_maxDown'] = 0
    data.loc[data.index[strt_idx: end_idx+1], 'in_maxDown'] = 1
    plot_series(data, {'close': '.-b'},
                cols_to_label_info={'close':
                            [['in_maxDown', (1,), ('.-r',), ('最大涨幅区间',)]]},
                title='最大涨幅：' + ' --> '.join(_) + ' (' + maxDown + ')',
                grids=True, figsize=(11, 7))

    #%%
    # 收益率、波动率、夏普测试
    fpath = '../test/510050_daily_pre_fq.csv'
    data = load_csv(fpath)
    data.set_index('date', drop=False, inplace=True)

    # 年化收益
    print('50ETF年化收益率：{}.'.format(round(cal_returns_period(data.close), 6)))

    # 波动率
    n = 90
    volt = round(cal_volatility(data.close.iloc[-n:], 'pct'), 6)
    print('50ETF {} 日年化收益波动率：{}.'.format(n if n >0 else None, volt))

    # 夏普
    n = 2000
    values = data.close.iloc[-n:]
    sharpe = round(cal_sharpe(values, r=3/100, nn=252, gain_type='log'), 6)
    print('50ETF {} 日年化夏普比率：{}.'.format(n if n >0 else None, sharpe))

    # beta
    df_beta = load_csv('../test/000001.XSHG_qfq.csv').set_index('time')
    df_beta = df_beta[['close']].rename(columns={'close': 'base'})
    tmp = load_csv('../test/600519.XSHG_qfq.csv').set_index('time')
    df_beta = pd.merge(df_beta, tmp[['close']], how='right',
                       left_index=True, right_index=True)
    df_beta = df_beta[df_beta.index >= '2006-01-01'].copy()
    df_beta.fillna(method='ffill', inplace=True)
    beta = cal_beta(df_beta['close'], df_beta['base'], gain_type='log')
    print('贵州茅台相对上证指数的beta值: {}'.format(beta))

    #%%
    # 信号盈亏统计
    data['signal'] = 0
    np.random.seed(1)
    for k in range(0, data.shape[0], 5):
        data.loc[data.index[k], 'signal'] = np.random.randint(-1, 2)

    sig_col = 'signal'
    func_vol_add = 'hold_1'
    # func_vol_add = lambda x, y, a, b: 2*x + 2*abs(y)
    # func_vol_sub = 'hold_1'
    # func_vol_sub = 'hold_base_1'
    func_vol_sub = lambda x, y, a, b, c: 2*x + 2*abs(y)
    func_vol_stoploss = 'hold_2'
    func_vol_stopgain = 'hold_2'
    ignore_no_stop = False
    col_price = 'close'
    col_price_buy = 'close'
    col_price_sel = 'close'
    # base_money = 15000
    base_money = None
    base_vol = 10000
    fee = 1.5/1000

    # max_loss, max_gain, max_down = None, None, None # 不设止盈止损
    max_loss, max_gain, max_down = 1.0/100, 1.5/100, None # 盈亏比止盈止损
    # max_loss, max_gain, max_down = None, None, 2.0/100 # 最大回撤止盈止损

    data = data.iloc[-50:, :].copy()
    trade_gain_info, df = cal_sig_gains(data, sig_col,
                                        func_vol_add=func_vol_add,
                                        func_vol_sub=func_vol_sub,
                                        func_vol_stoploss=func_vol_stoploss,
                                        func_vol_stopgain=func_vol_stopgain,
                                        ignore_no_stop=ignore_no_stop,
                                        col_price=col_price,
                                        col_price_buy=col_price_buy,
                                        col_price_sel=col_price_sel,
                                        base_money=base_money, base_vol=base_vol,
                                        fee=fee, max_loss=max_loss,
                                        max_gain=max_gain, max_down=max_down,
                                        del_begin0=False, gap_repeat=False)
    plot_series(df, {'close': ('.-k', False)},
                cols_to_label_info={'close':
                [['act', (-1, 1), ('r^', 'bv'), ('做多', '做空')],
                 ['act_stop', (-1.5, 1.5, -0.5, 0.5), ('r*', 'b*', 'mo', 'go'),
                  ('做空止盈', '做多止盈', '做空止损', '做多止损')]
                ]},
                markersize=15, figsize=(11, 7))
    pprint(trade_gain_info)


    # test1
    data_ = pd.DataFrame({'close': [100, 80, 300, 400, 900, 1000, 995, 990,
                                    900, 800],
                          'signal': [1, -1, 0, 0, 0, 0, 0, 0, 0, 0]})
    trade_gain_info, df_ = cal_sig_gains(data_, sig_col,
                                         base_money=None, base_vol=1,
                                         max_loss=None, max_gain=None, fee=0,
                                         max_down=None, func_vol_sub='hold_3')
    plot_series(df_, {'close': ('.-k', False)},
                cols_styl_up_right={'AccountValue_maxUsed_net': '-r'},
                cols_to_label_info={'close':
                [['act', (-1, 1), ('r^', 'bv'), ('做多', '做空')],
                  ['act_stop', (-1.5, 1.5, -0.5, 0.5), ('r*', 'b*', 'mo', 'go'),
                  ('做空止盈', '做多止盈', '做空止损', '做多止损')]
                ]},
                markersize=15, figsize=(11, 7))

    # test2
    data_ = pd.DataFrame({'close': [100, 200, 200, 200, 400, 100, 100, 200,
                                    300, 100, 50],
                          'signal': [0, 0, 1, 0, -1, 1, 0, 0, 0, 0, 0]})
    trade_gain_info, df_gain = get_yield_curve(
                                data_, sig_col,
                                base_money=None, base_vol=1, init_cash=5000,
                                max_loss=None, max_gain=None, fee=10/100,
                                max_down=None, func_vol_sub='base_2',
                                stop_sig_order='both',
                                force_final0='settle')

    # pprint(trade_gain_info)

    # test3
    data__ = pd.DataFrame({'close': [100, 200, 200, 200, 400, 100, 100, 200,
                                      300, 100, 50],
                          'trade_vol': [0, 0, -1, 0, 2, -2, 0, 0, 0, 0, 0]})
    trade_gain_info_, df_gain_ = get_yield_curve(
                                    data__, 'trade_vol', sig_type=2,
                                    base_money=None, base_vol=1, init_cash=50,
                                    max_loss=None, max_gain=None, fee=10/100,
                                    max_down=None, stop_sig_order='both',
                                    force_final0='settle')

    pprint(trade_gain_info_)

    # test4
    data4 = pd.DataFrame({'close': [100, 200, 200, 200, 400, 100, 100, 200],
                          'trade_vol': [3, 5, 4, -5, -3, 0, -4, 1]})
    trade_gain_info4, df_gain4 = get_yield_curve(
                                    data4, 'trade_vol', sig_type=2,
                                    base_money=None, base_vol=1, init_cash=50,
                                    max_loss=None, max_gain=None, fee=10/100,
                                    max_down=None, stop_sig_order='both',
                                    force_final0='settle')

    # pprint(trade_gain_info4)

    # test5
    data5 = pd.DataFrame({'close': [100, 105, 500, 105],
                          'trade_vol': [-1, 0, 1, 0]})
    trade_gain_info5, df_gain5 = get_yield_curve(
                                    data5, 'trade_vol', sig_type=2,
                                    base_money=None, base_vol=1, init_cash=300,
                                    max_loss=None, max_gain=None, fee=0/100,
                                    max_down=None, stop_sig_order='both',
                                    sos_money=10, force_final0='settle')

    # pprint(trade_gain_info5)

    # test6
    data6 = pd.DataFrame({'close': [100, 105, 100, 105],
                          'trade_vol': [1, 0, -2, 0]})
    trade_gain_info6, df_gain6 = get_yield_curve(
                                    data6, 'trade_vol', sig_type=2,
                                    base_money=None, base_vol=1, init_cash=0,
                                    max_loss=None, max_gain=None, fee=0/100,
                                    max_down=None, stop_sig_order='both',
                                    sos_money=10, force_final0='settle')

    # pprint(trade_gain_info6)

    # test7
    data7 = pd.DataFrame({'close': [100, 95, 89, 85, 70, 60, 90, 105],
                          'signal': [1, 0, -1, 0, 0, 0, 0, 0]})
    trade_gain_info7, df_gain7 = get_yield_curve(
                                    data7, 'signal', sig_type=1,
                                    base_money=None, base_vol=1, init_cash=0,
                                    max_loss=None, max_gain=None, fee=0/100,
                                    max_down=None, stop_sig_order='both',
                                    add_sig_order='sig_only', add_gain_pct=10/100,
                                    add_loss_pct=10/100,
                                    sos_money=10, force_final0='settle')

    #%%
    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
