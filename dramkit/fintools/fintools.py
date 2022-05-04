# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from dramkit.plottools.plot_common import plot_series
from dramkit.gentools import x_div_y, isnull
from dramkit.datsci.stats import avedev
from dramkit.gentools import replace_repeat_pd
from dramkit.datsci.stats import cum_scale, cum_pct_loc
from dramkit.datsci.stats import rolling_scale, rolling_pct_loc

try:
    import talib
except:
    from dramkit.logtools.logger_general import get_logger
    from dramkit.logtools.utils_logger import logger_show
    logger = get_logger()
    logger_show('导入talib失败，部分指标计算功能将无法使用！',
                logger, 'warn')


def cal_IC(data, col_factor, col_price='close', col_time=None,
           col_code=None, lag=1, ic_type='rank'):
    '''
    | 计算因子IC值
    | col_factor、col_price、col_time、col_code分别为因子列、价格列、时间列、代码列
    | lag设置期数，ic_type设置IC类型，可选`normal`和`rank`
    '''
    df = data.copy()
    if isnull(col_time):
        df.index.name = 'T'
        df.reset_index(drop=False, inplace=True)
        col_time = 'T'
    if isnull(col_code):
        df['code'] = 'code0'
        col_code = 'code'
    df = df.reindex(columns=[col_code, col_time, col_price, col_factor])
    df_ = []
    def _get_pct(df):
        df.sort_values(col_time, inplace=True)
        df['pct'] = df[col_price].shift(-lag) / df[col_price] - 1
        return df
    for code in df[col_code].unique():
        df_.append(_get_pct(df[df[col_code] == code].copy()))
    df = pd.concat(df_, axis=0)
    if ic_type == 'normal':
        return df[[col_factor, 'pct']].corr().loc[col_factor, 'pct']
    elif ic_type == 'rank':
        return df[[col_factor, 'pct']].corr(method='spearman').loc[col_factor, 'pct']


def kelly_formula(p, b):
    '''
    | 凯利公式：根据胜率p和盈亏比b计算最优仓位
    | 参考：
    | https://zhuanlan.zhihu.com/p/38279377
    | https://xueqiu.com/6551266259/146636046
    '''
    q = 1- p
    return (b*p - q) / b


def sim_MC(his_values, sim_periods, n_cut=240, num_sim=1000,
           random_seed=None, keep_v0=True, return_mid=False,
           kwargs_plot_mid=None, kwargs_plot_final=None):
    '''
    蒙特卡洛法模拟序列数据变化路径

    Parameters
    ----------
    his_values : pd.Series, np.array, list
        历史序列
    sim_periods : int
        向后模拟期数
    n_cut : int
        将一期时间划分为多少个小时段
    num_sim : int
        模拟次数
    random_seed : None, int
        随机数种子设置
    keep_v0 : bool
        模拟结果中是否保留第一个初始价格
    return_mid : bool
        是否范围模拟过程中的中间路径数据
    kwargs_plot_mid : dict
        中间路径数据绘图参数， ``plot_series`` 函数接收
    kwargs_plot_final : dict
        最终模拟数据绘图参数， ``plot_series`` 函数接收

    Returns
    -------
    df_sim : pd.DataFrame
        模拟结果，shape为(sim_periods, num_sim)，即每列为一条模拟路径
    df_sim_ : pd.DataFrame
        包含中间路径的模拟结果， ``return_mid`` 为True时返回

    References
    ----------
    https://zhuanlan.zhihu.com/p/342624971
    '''

    np.random.seed(random_seed)

    df = pd.DataFrame({'p': his_values})
    df['pcts'] = df['p'].pct_change() # 历史变化率序列
    mu = df['pcts'].mean() # 变化率均值
    sigma = df['pcts'].std() # 波动率

    deltaT = 1 / n_cut
    deltaT_sqrt = np.sqrt(deltaT)
    mu_deltaT = mu * deltaT
    sigma_deltaT_sqrt = sigma * deltaT_sqrt
    sim_periods_ = sim_periods * n_cut

    P0 = pd.Series(his_values).iloc[-1]
    df_sim_ = pd.DataFrame({'P0': np.ones(num_sim) * P0})
    for k in range(1, sim_periods_+1):
        df_sim_['P{}'.format(k)] = np.random.randn(num_sim)
        df_sim_['P{}'.format(k)] = df_sim_['P{}'.format(k-1)] + df_sim_['P{}'.format(k-1)] * \
                           (mu_deltaT + df_sim_['P{}'.format(k)] * sigma_deltaT_sqrt)
    if keep_v0:
        cols = ['P{}'.format(k*n_cut) for k in range(0, sim_periods+1)]
    else:
        cols = ['P{}'.format(k*n_cut) for k in range(1, sim_periods+1)]
    if not kwargs_plot_mid is None:
        tmp = df_sim_.transpose()
        plot_series(tmp, {x: ('-', False) for x in tmp.columns},
                    **kwargs_plot_mid)
    df_sim = df_sim_.reindex(columns=cols)
    df_sim = df_sim.transpose()
    df_sim.reset_index(drop=True, inplace=True)
    df_sim.columns = ['sim_{}'.format(k) for k in range(1, df_sim.shape[1]+1)]
    if not kwargs_plot_final is None:
        plot_series(df_sim, {x: ('-', False) for x in df_sim.columns},
                    **kwargs_plot_final)

    if return_mid:
        return df_sim, df_sim_
    else:
        return df_sim


def wuxianpu(series, lag=875, alphas=[0.05, 0.32], plot=False):
    '''
    | 五线谱指标
    | 参考：
    | https://caibaoshuo.com/guides/15
    '''
    def get_reg_results(x):
        X = np.arange(0, len(x)).reshape(-1, 1)
        X = sm.add_constant(X)
        y =  np.array(x)
        mdl = sm.OLS(y, X)
        results = mdl.fit()
        y_pre = results.fittedvalues
        _, ypre_low1, ypre_up1 = wls_prediction_std(results, alpha=alphas[1])
        _, ypre_low2, ypre_up2 = wls_prediction_std(results, alpha=alphas[0])
        if plot:
            tmp = pd.DataFrame({'x': x, 'y_pre': y_pre,
                                'ypre_low1': ypre_low1, 'ypre_low2': ypre_low2,
                                'ypre_up1': ypre_up1, 'ypre_up2': ypre_up2})
            tmp.index = x.index
            plot_series(tmp, {'x': '-k', 'y_pre': '-b',
                              'ypre_low1': '-m', 'ypre_low2': '-r',
                              'ypre_up1': '-c', 'ypre_up2': '-g'})
        return y_pre[-1], ypre_low1[-1], ypre_up1[-1], ypre_low2[-1], ypre_up2[-1]
    df = pd.DataFrame({'p': series})
    ori_idx = df.index
    df.index = range(0, df.shape[0])
    df['ypre'] = np.nan
    df['ypre_low1'] = np.nan
    df['ypre_low2'] = np.nan
    df['ypre_up1'] = np.nan
    df['ypre_up2'] = np.nan
    for k in range(lag, df.shape[0]):
        x = df['p'].iloc[k-lag:k]
        x.index = ori_idx[k-lag:k]
        y_pre, ypre_low1, ypre_up1, ypre_low2, ypre_up2 = get_reg_results(x)
        df.loc[df.index[k], 'y_pre'] = y_pre
        df.loc[df.index[k], 'ypre_low1'] = ypre_low1
        df.loc[df.index[k], 'ypre_low2'] = ypre_low2
        df.loc[df.index[k], 'ypre_up1'] = ypre_up1
        df.loc[df.index[k], 'ypre_up2'] = ypre_up2
    df.index = ori_idx
    df = df.reindex(columns=['y_pre', 'ypre_low1', 'ypre_low2', 'ypre_up1',
                             'ypre_up2'])
    return df


def MAs_order(series, lags=[3, 5, 15, 20, 30, 60]):
    '''
    | 指定均线排列情况判断，lags为从小到大排列的lag列表
    | 返回结果1为多头排列，-1位空头排列，0其他情况
    '''
    df = pd.DataFrame({'p': series})
    for lag in lags:
        df['ma_{}'.format(lag)] = df['p'].rolling(lag).mean()
    def _get_order_type(x):
        order_type = 0
        for k in range(1, len(lags)):
            if x['ma_{}'.format(lags[k])] >= x['ma_{}'.format(lags[k-1])]:
                order_type -= 1
            elif x['ma_{}'.format(lags[k])] <= x['ma_{}'.format(lags[k-1])]:
                order_type += 1
        if order_type == len(lags) - 1:
            return 1
        elif order_type == 1 - len(lags):
            return -1
        else:
            return 0
    df['order_type'] = df[['ma_{}'.format(lag) for lag in lags]].apply(
                                    lambda x: _get_order_type(x), axis=1)
    return df['order_type']


def MACD(series, fast=12, slow=26, m=9):
    '''
    MACD计算
    
    Parameters
    ----------
    series : pd.Series, np.array, list
        待计算序列
    fast : int
        短期窗口长度
    slow : int
        长期窗口长度
    m : int
        平滑移动平均窗口长度
    
    Returns
    -------
    df : pd.DataFrame
        包含['MACD', 'DIF', 'DEA']三列
    
    Note
    ----
    计算结果与同花顺PC统一版基本能对上，但是跟远航版没对上
    
    References
    ----------
    - http://www.360doc.com/content/17/1128/12/50117541_707746936.shtml
    - https://baijiahao.baidu.com/s?id=1602850251881203999&wfr=spider&for=pc
    - https://www.cnblogs.com/xuruilong100/p/9866338.html
    - https://blog.csdn.net/u012724887/article/details/105358115
    '''

    col = 'series'
    series.name = col
    df = pd.DataFrame(series)

    # df['DI'] = (df['high'] + df['close'] + 2*df['low']) / 4
    df['DI'] = df[col].copy()

    df['EMA_fast'] = df['DI'].copy()
    df['EMA_slow'] = df['DI'].copy()
    k = 1
    while k < df.shape[0]:
        df.loc[df.index[k], 'EMA_fast'] = \
                       df.loc[df.index[k], 'DI'] * 2 / (fast+1) + \
                       df.loc[df.index[k-1], 'EMA_fast'] * (fast-1) / (fast+1)
        df.loc[df.index[k], 'EMA_slow'] = \
                       df.loc[df.index[k], 'DI'] * 2 / (slow+1) + \
                       df.loc[df.index[k-1], 'EMA_slow'] * (slow-1) / (slow+1)
        k += 1
    df['DIF'] = df['EMA_fast'] - df['EMA_slow']
    df['DEA'] = df['DIF'].copy()
    k = 1
    while k < df.shape[0]:
        df.loc[df.index[k], 'DEA'] = \
                                df.loc[df.index[k], 'DIF'] * 2 / (m+1) + \
                                df.loc[df.index[k-1], 'DEA'] * (m-1) / (m+1)
        k += 1
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])

    return df[['MACD', 'DIF', 'DEA']]


def Boll(series, lag=15, width=2, n_dot=3):
    '''
    布林带计算

    Parameters
    ----------
    series : pd.Series
        序列数据
    lag : int
        历史期数
    width : int, float
        计算布林带上下轨时用的标准差倍数（宽度）
    n_dot : int
        计算结果保留小数位数

    Returns
    -------
    df_boll : pd.DataFrame
        布林带，包含原始值(即series)，布林带下轨值(boll_low列)，
        中间值(boll_mid列)，上轨值(boll_up列)，标准差(std列)
    '''

    df_boll = pd.DataFrame(series)
    col = df_boll.columns[0]
    df_boll['boll_mid'] = df_boll[col].rolling(lag).mean()
    df_boll['boll_std'] = df_boll[col].rolling(lag).std()
    df_boll['boll_up'] = df_boll['boll_mid'] + width*df_boll['boll_std']
    df_boll['boll_low'] = df_boll['boll_mid'] - width*df_boll['boll_std']

    df_boll['boll_mid'] = df_boll['boll_mid'].round(n_dot)
    df_boll['boll_up'] = df_boll['boll_up'].round(n_dot)
    df_boll['boll_low'] = df_boll['boll_low'].round(n_dot)

    df_boll = df_boll.reindex(columns=[col, 'boll_low', 'boll_mid', 'boll_up',
                                       'boll_std'])

    return df_boll


def BBI(series, lags=[3, 6, 12, 24]):
    '''BBI计算'''
    df = pd.DataFrame({'p': series})
    df['bbi'] = 0
    for lag in lags:
        df['bbi'] = df['bbi'] + df['p'].rolling(lag).mean()
    df['bbi'] = df['bbi'] / len(lags)
    return df['bbi']


def CCI(df, col_typeprice=None, n=14, r=0.015):
    '''
    CCI计算

    Parameters
    ----------
    df : pd.DataFrame
        历史行情数据，须包含['high', 'low', 'close']列或col_typeprice列
    col_typeprice : None, str
        用于计算CCI的价格列，若不指定，则根据['high', 'low', 'close']计算
    n : int
        计算周期
    r : float
        计算系数

    Returns
    -------
    cci : pd.Series
        CCI序列

    References
    ----------
    - 同花顺PC端CCI指标公式
    - https://blog.csdn.net/spursping/article/details/104485136
    - https://blog.csdn.net/weixin_43055882/article/details/86696954
    '''

    if col_typeprice is not None:
        if col_typeprice in df.columns:
            df_ = df.reindex(columns=[col_typeprice])
        else:
            raise ValueError('请检查col_typeprice列名！')
    else:
        df_ = df.reindex(columns=['high', 'low', 'close'])
        df_[col_typeprice] = df_[['high', 'low', 'close']].sum(axis=1) / 3

    df_['MA'] = df_[col_typeprice].rolling(n).mean()
    df_['MD'] = df_[col_typeprice].rolling(n).apply(lambda x: avedev(x))

    cci = (df_[col_typeprice] - df_['MA']) / (df_['MD'] * r)

    return cci


def EXPMA(series, n=26):
    '''
    EXPMA计算

    References
    ----------
    - https://blog.csdn.net/zxyhhjs2017/article/details/93499930
    - https://blog.csdn.net/ydjcs567/article/details/62249627
    - tradingview公式
    '''
    r = 2 / (n + 1)
    df_ = pd.DataFrame({'p': series})
    df_['expma'] = df_['p']
    for k in range(1, df_.shape[0]):
        x0 = df_.loc[df_.index[k-1], 'expma']
        x = df_.loc[df_.index[k], 'p']
        df_.loc[df_.index[k], 'expma'] = r * (x - x0) + x0
    return df_['expma']


def weight_MA_linear_decay(series, n=15):
    '''加权MA，权重呈线性递减'''
    return series.rolling(n).apply(
           lambda x: np.average(x, weights=list(range(1, len(x)+1))))


def KAMA(series, lag=9, fast=2, slow=30):
    '''
    Kaufman's Adaptive Moving Average (KAMA)
    
    References
    ----------
    - https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average
    - https://www.technicalindicators.net/indicators-technical-analysis/152-kama-kaufman-adaptive-moving-average
    '''

    col = 'series'
    series.name = col
    df = pd.DataFrame(series)

    df['direction'] = abs(df[col].diff(periods=lag))
    df['diff'] = abs(df[col].diff())
    df['volatility'] = df['diff'].rolling(lag).sum()
    df['ER'] = df[['direction', 'volatility']].apply(lambda x:
                    x_div_y(x['direction'], x['volatility'], v_y0=0), axis=1)

    fast_alpha = 2 / (fast + 1)
    slow_alpha = 2 / (slow + 1)
    df['SC'] = (df['ER'] * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    df['kama'] = np.nan
    k = lag
    df.loc[df.index[k], 'kama'] = (df[col].iloc[:k+1]).mean()
    k += 1
    while k < df.shape[0]:
        pre_kama = df.loc[df.index[k-1], 'kama']
        sc = df.loc[df.index[k], 'SC']
        price = df.loc[df.index[k], col]
        df.loc[df.index[k], 'kama'] = pre_kama + sc * (price - pre_kama)
        k += 1

    return df['kama']


def ER(series, lag=10, sign=False):
    '''
    价格效率
    
    References
    ----------
    - KAMA指标
    - 华泰联合-震荡市还是趋势市：市场状态的量化划分方法
    '''
    df = pd.DataFrame({'p': series})
    if sign:
        df['dirt_chg'] =  df['p'].diff(periods=lag)
    else:
        df['dirt_chg'] = abs(df['p'].diff(periods=lag))
    df['acum_chg'] = abs(df['p'].diff()).rolling(lag).sum()
    df['er'] = df[['dirt_chg', 'acum_chg']].apply(lambda x:
               x_div_y(x['dirt_chg'], x['acum_chg'], v_y0=0), axis=1)
    return df['er']


def ma_trend_strength(series, lag_fast=5, lag_slow=20):
    '''
    用均线度量的市场趋势强度
    
    References
    ----------
    华泰联合-震荡市还是趋势市：市场状态的量化划分方法
    '''
    df = pd.DataFrame({'p': series})
    df['ma_fast'] = df['p'].rolling(lag_fast).mean()
    df['ma_slow'] = df['p'].rolling(lag_slow).mean()
    df['ma_fast_'] = df['ma_fast'].shift(1)
    df['strength1'] = abs((df['ma_fast'] - df['ma_slow']) / df['ma_slow'])
    df['strength2'] = abs((df['ma_fast'] - df['ma_fast_']) / df['ma_fast'])
    df['strength'] = df['strength1'] + df['strength2']
    return df['strength']


def KDJ(data, n_his=9, n_k=3, n_d=3, n_k_=1, n_d_=1):
    '''
    | KDJ计算，data(pd.DataFrame)中必须包含['close', 'high', 'low']三列
    | 返回结果中包含=['K', 'D', 'J', 'RSV']四列
    | 注：计算结果跟同花顺能对上
    | 参考：
    | tradingview公式
    | https://blog.csdn.net/qq337484627/article/details/110727392
    '''
    df = data.reindex(columns=['close', 'high', 'low'])
    df['high_'] = df['high'].rolling(n_his).max()
    df['low_'] = df['low'].rolling(n_his).min()
    df['RSV'] = 100 * ((df['close']-df['low_']) / (df['high_']-df['low_']))
    df['K'], df['D'] = 0, 0
    for k in range(n_his-1, df.shape[0]):
        RSV_now = df.loc[df.index[k], 'RSV']
        K_pre = df.loc[df.index[k-1], 'K']
        K_now = (n_k_*RSV_now + (n_k-n_k_)*K_pre) / n_k
        df.loc[df.index[k], 'K'] = K_now
        D_pre = df.loc[df.index[k-1], 'D']
        D_now = (n_d_*K_now + (n_d-n_d_)*D_pre) / n_d
        df.loc[df.index[k], 'D'] = D_now
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df.reindex(columns=['K', 'D', 'J', 'RSV'])


def talib_RSI(series, n=20):
    '''
    talib计算RSI
    '''
    return talib.RSI(series, n)


def ATR(df, lag=14):
    '''
    | 平均真实波幅ATR计算，df中必须包含['high', 'low', 'close']三列
    | 返回ATR, TR
    '''
    df['close_pre'] = df['close'].shift(1)
    df['TR1'] = df['high'] - df['low']
    df['TR2'] = abs(df['close_pre'] - df['high'])
    df['TR3'] = abs(df['close_pre'] - df['low'])
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(lag).mean()
    return df['ATR'], df['TR']


def talib_ATR(df, lag=14):
    '''
    | talib计算平均真实波幅ATR
    | df中必须包含['high', 'low', 'close']三列
    | 注：talib的结果跟同花顺对不上
    | 参考：
    | https://www.bilibili.com/read/cv7141776/
    '''
    return talib.ATR(df['high'], df['low'], df['close'], lag)


def ADX(df, lag=14):
    '''
    | 平均趋向指标ADX计算
    | df中必须包含['high', 'low', 'close']三列
    | 参考：
    | https://zhuanlan.zhihu.com/p/64827704
    '''
    raise NotImplementedError


def talib_ADX(df, lag=14):
    '''
    | talib计算平均趋向指标ADX
    | df中必须包含['high', 'low', 'close']三列
    '''
    return talib.ADX(df['high'], df['low'], df['close'], lag)


def talib_ROC(series, lag=12):
    '''talib计算ROC'''
    return talib.ROC(series, timeperiod=lag)


def ROC(series, lag=12):
    '''ROC计算'''
    df = pd.DataFrame({'p': series})
    df['roc'] = 100 * df['p'].pct_change(lag)
    return df['roc']


def VRI(data, n=3):
    '''
    | VRI计算
    | 参考：
    | https://finquanthub.com/量化策略：基于波动率范围指标的反向交易策略/
    '''
    df = data.reindex(columns=['close', 'high', 'low'])
    df['volatility'] = df['close'].rolling(n).std()
    df['momentum'] = df['close'].diff(n)
    df['ext_range'] = df['high'].rolling(n).max() - df['low'].rolling(n).min()
    df['vri'] = ((df['momentum'] / df['ext_range']) * df['volatility'])
    df['vri'] = df['vri']
    return df['vri']


def BBW(series, lag=20, width=2, sign=False):
    '''BBW计算'''
    df = pd.DataFrame(series)
    col = df.columns[0]
    df['mid'] = df[col].rolling(lag).mean()
    df['std'] = df[col].rolling(lag).std()
    df['bbw'] = 2 * df['std'] / df[col]
    if sign:
        df['sign'] = df[col] - df['mid']
        df['sign'] = df['sign'].apply(lambda x: 1 if x > 0 else \
                                      (-1 if x < 0 else 0))
        df['bbw'] = df['bbw'] * df['sign']
    return df['bbw']


def mean_candle(data):
    '''
    | 平均K线计算
    | data(pd.DataFrame)中必须包含['open', 'close', 'high', 'low']列
    | 参考：
    | https://cn.tradingview.com/script/YfPnKE22-Heikin-Ashi-MACD/
    '''

    df = data.reindex(columns=['open', 'high', 'low', 'close'])

    df['close_m'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['open_m'] = (df['open'] + df['close']) / 2
    for k in range(1, df.shape[0]):
        df.loc[df.index[k], 'open_m'] = (df.loc[df.index[k-1], 'open_m'] + \
                                         df.loc[df.index[k-1], 'close_m']) / 2
    df['high_m'] = df[['high', 'close_m', 'open_m']].max(axis=1)
    df['low_m'] = df[['low', 'close_m', 'open_m']].min(axis=1)

    df = df.reindex(columns=['open_m', 'high_m', 'low_m', 'close_m'])
    df.rename(columns={'open_m': 'open', 'close_m': 'close',
                       'high_m': 'high', 'low_m': 'low'}, inplace=True)

    return df


def DeMark_TD(series, n=9, lag=4):
    '''
    | 迪马克TD序列：连续n次出现收盘价高/低于前面第lag个收盘价就发信号
    | 默认九转序列
    | 返回label序列，label取1表示高点信号，-1表示低点信号
    '''

    if series.name is None:
        series.name = 'series'
    col = series.name
    df = pd.DataFrame(series)

    # 是否大于|小于前面第lag个信号
    df[col+'_preLag'] = df[col].shift(lag)
    df['dif_preLag'] = df[col] - df[col+'_preLag']
    df['dif_preLag'] = df['dif_preLag'].apply(lambda x:
                                        1 if x > 0 else (-1 if x < 0 else 0))

    # 计数
    df['count'] = 0
    k = 0
    while k < df.shape[0]:
        if df.loc[df.index[k], 'dif_preLag'] == 0:
            k += 1
        elif df.loc[df.index[k], 'dif_preLag'] == 1:
            label = 1
            df.loc[df.index[k], 'count'] = label
            ktmp = k + 1
            while ktmp < df.shape[0] and \
                                df.loc[df.index[ktmp], 'dif_preLag'] == 1:
                if label == n:
                    label = 1
                else:
                    label += 1
                df.loc[df.index[ktmp], 'count'] = label
                ktmp += 1
            k = ktmp
        else:
            label = -1
            df.loc[df.index[k], 'count'] = label
            ktmp = k + 1
            while ktmp < df.shape[0] and \
                                df.loc[df.index[ktmp], 'dif_preLag'] == -1:
                if label == -n:
                    label = -1
                else:
                    label -= 1
                df.loc[df.index[ktmp], 'count'] = label
                ktmp += 1
            k = ktmp

    # 提取有效信号
    df['label'] = df['count'].apply(lambda x: 1 if x == n else \
                                    (-1 if x == -n else 0))

    return df['label']


def ACPMP(values, periods=range(2, 60), mode='sum', std_type='pct',
          lag_roll=None, std_order=False):
    '''
    多周期平均累计百分比(Average Cumulative Percentage of Multi-Period)
    
    Parameters
    ----------
    values : pd.Series, np.array, list
        待计算序列数据
    periods : list
        指定多个周期列表
    mode : str
        | 设置动量累计模式：
        | - 为`sum`时累计涨跌幅为周期内一期涨跌幅的累计求和
        | - 为`dirt`时累计涨跌幅为周期内最后一期相较于第一期的涨跌幅
    std_type : str
        | 设置滚动标准化方式：
        | - 若为`01`，则用maxmin方法转化到0-1之间
        | - 若为`pct`，则转化为百分位
    lag_roll : None, int
        设置为整数时，标准化时采用滚动方式而不是累计方式
    std_order : bool, None, str
        | 设置标准化顺序：
        | - 若为`first`，则先对每个周期进行标准化后再求平均
        | - 若为`last`，则先求平均之后再进行标准化
        | - 若为None或False，则不标准化
        
    Returns
    -------
    acpmp : pd.Series
        ACPMP计算结果序列
    '''

    if mode not in ['sum', 'dirt']:
        raise ValueError('mode只能设置为`sum`或`dirt`！')
    if std_type not in ['01', 'pct']:
        raise ValueError('std_type只能设置为`01`或`pct`！')
    if std_type not in ['01', 'pct']:
        raise ValueError('std_order只能设置为`first`或`last`或None|False！')

    df = pd.DataFrame({'v': values})

    if mode == 'sum':
        df['pct'] = df['v'].pct_change()
        for n in periods:
            df['acc_pct_{}'.format(n)] = df['pct'].rolling(n).sum()
    elif mode == 'dirt':
        for n in periods:
            df['acc_pct_{}'.format(n)] = df['v'].pct_change(n)

    if isnull(std_order) or not std_order:
        df['acpmp'] = df[['acc_pct_{}'.format(n) for n in periods]].mean(axis=1)
    else:
        if std_type == '01':
            if isnull(lag_roll) or not lag_roll:
                stdFunc = cum_scale
            else:
                stdFunc = lambda x: rolling_scale(x, lag_roll)
        elif std_type == 'pct':
            if isnull(lag_roll) or not lag_roll:
                stdFunc = cum_pct_loc
            else:
                stdFunc = lambda x: rolling_pct_loc(x, lag_roll)

        if std_order == 'first':
            for n in periods:
                df['acc_pct_{}_std'.format(n)] = stdFunc(df['acc_pct_{}'.format(n)])
            df['acpmp'] = df[['acc_pct_{}_std'.format(n) for n in periods]].mean(axis=1)
        elif std_order == 'last':
            df['acpmp'] = df[['acc_pct_{}'.format(n) for n in periods]].mean(axis=1)
            df['acpmp'] = stdFunc(df['acpmp'])

    return df['acpmp']


def get_turn_point(series):
    '''
    | 拐点标注
    | 返回结果中1位从上往下拐，-1位从下往上拐
    '''
    df = pd.DataFrame({'v': series})
    df['dif'] = df['v'].diff()
    df['turn'] = df['dif'].apply(lambda x: 1 if x > 0 else \
                                                     (-1 if x < 0 else 0))
    for k in range(df.shape[0]-1):
        if df['dif'].iloc[k] == 0 and df['turn'].iloc[k] == 0:
            df.loc[df.index[k], 'turn'] = -1 * df['turn'].iloc[k+1]
        else:
            continue
    df['turn'] = df['turn'] * df['turn'].shift(1)
    df['turn'] = df[['turn', 'dif']].apply(lambda x:
                -1 if x['turn'] == -1 and x['dif'] > 0 else \
                (1 if x['turn'] == -1 and x['dif'] < 0 else 0), axis=1)
    return df['turn']


def cross(series, series_base):
    '''
    | 生成交叉信号（series上穿或下穿series_base）
    | 返回信号中-1为上穿信号，1为下穿信号
    '''
    series_dif = series - series_base
    df = pd.DataFrame({'vdif': series_dif})
    df['cross'] = df['vdif'].apply(lambda x:
                                       -1 if x > 0 else (1 if x < 0 else 0))
    df['cross'] = replace_repeat_pd(df['cross'], 1, 0)
    df['cross'] = replace_repeat_pd(df['cross'], -1, 0)
    k = 0
    while k < df.shape[0] and df.loc[df.index[k], 'cross'] == 0:
        k += 1
    df.loc[df.index[k], 'cross'] = 0
    return df['cross']


def cross_plot(data, col, col_base, col_price='close',
               plot=True, plot_down=True, plot_same_level=False,
               **kwargs_plot):
    '''在 :func:`dramkit.fintools.fintools.cross` 基础上增加画图'''
    df = data.copy()
    if isinstance(col_base, str) and col_base in df.columns:
        df['col_base_'] = df[col_base].copy()
    else:
        df['col_base_'] = col_base
    df['signal'] = cross(df[col], df['col_base_'])
    if plot:
        if plot_down:
            plot_series(df, {col_price: '.-k'},
                        cols_styl_low_left={col: '-c',
                                            'col_base_': ('-b', col_base)},
                        cols_to_label_info={
                            col_price:
                                [['signal', (-1, 1), ('r^', 'gv'), False]],
                            col:
                                [['signal', (-1, 1), ('r^', 'gv'), False,
                                  {'markersize': 8}]]
                            },
                        **kwargs_plot)
        else:
            if plot_same_level:
                plot_series(df, {col_price: '.-k',
                                 col: '-c', 'col_base_': ('-b', col_base)},
                            cols_to_label_info={
                                col_price:
                                    [['signal', (-1, 1), ('r^', 'gv'), False]],
                                # col:
                                #     [['signal', (-1, 1), ('r.', 'g.'), False]]
                                    },
                            **kwargs_plot)
            else:
                plot_series(df, {col_price: '.-k'},
                            cols_styl_up_right={'col_base_': ('-b', col_base),
                                                col: '-c',},
                            cols_to_label_info={
                                col_price:
                                    [['signal', (-1, 1), ('r^', 'gv'), False]],
                                col:
                                    [['signal', (-1, 1), ('r.', 'g.'), False]]
                                    },
                            **kwargs_plot)
            plot_series(df, {col_price: '.-k'},
                    cols_to_label_info={col_price:
                                  [['signal', (-1, 1), ('r^', 'gv'), False]]},
                    **kwargs_plot)
    return df['signal']


def cross2(series_buy, base_buy, series_sel, base_sel,
           buy_sig=1, sel_sig=-1):
    '''
    | 生成由两个交叉信号最终合成的买卖信号
    | series_buy和base_buy交叉生成买入信号:
    |     - 当buy_sig=-1时以series_buy上穿base_buy为买入信号
    |     - 当buy_sig=1时以series_buy下穿base_buy为买入信号
    | series_sel和base_sel交叉生成卖出信号:
    |     - 当buy_sel=-1时以series_sel上穿base_sel为卖出信号
    |     - 当buy_sel=1时以series_sel下穿base_sel为卖出信号
    | 返回值中-1为买，1为卖，没信号
    '''
    if buy_sig not in [1, -1] or sel_sig not in [1, -1]:
        raise ValueError('buy_sig和sel_sig必须设置为-1或1')
    crs_buy = cross(series_buy, base_buy)
    crs_sel = cross(series_sel, base_sel)
    df = pd.DataFrame({'crs_buy': crs_buy, 'crs_sel': crs_sel})
    df['sig_buy'] = df['crs_buy'].apply(lambda x:
                                        -1 if buy_sig == -1 and x == -1 else \
                                        (-1 if buy_sig == 1 and x == 1 else 0))
    df['sig_sel'] = df['crs_sel'].apply(lambda x:
                                        1 if sel_sig == -1 and x == -1 else \
                                        (1 if sel_sig == 1 and x == 1 else 0))
    df['signal'] = df['sig_buy'] + df['sig_sel']
    return df['signal']


def cross2_plot(data, col_buy, base_buy, col_sel, base_sel,
                buy_sig=1, sel_sig=-1, col_price='close',
                plot=True, **kwargs_plot):
    '''在 :func:`cross2` 函数的信号基础上增加绘图'''
    df = data.copy()
    if isinstance(base_buy, str) and base_buy in df.columns:
        df['base_buy'] = df[base_buy].copy()
    else:
        df['base_buy'] = base_buy
    if isinstance(base_sel, str) and base_sel in df.columns:
        df['base_sel'] = df[base_sel].copy()
    else:
        df['base_sel'] = base_sel
    df['signal'] = cross2(df[col_buy], df['base_buy'],
                          df[col_sel], df['base_sel'],
                          buy_sig=buy_sig, sel_sig=sel_sig)
    if plot:
        if col_buy == col_sel:
            plot_series(df, {col_price: '.-k'},
                        cols_styl_up_right={'base_buy': ('--b', False),
                                            'base_sel': ('--b', False),
                                            col_buy: '.-b'},
                        cols_to_label_info={
                         col_price: [['signal', (-1, 1), ('r^', 'gv'), False]],
                         col_buy: [['signal', (-1, 1), ('r.', 'g.'), False]]},
                        **kwargs_plot)
        else:
            plot_series(df, {col_price: '.-k'},
                        cols_styl_up_right={'base_buy': ('--b', False),
                                            col_buy: '.-b'
                                            },
                        cols_to_label_info={
                            col_price: [['signal', (-1,), ('r^',), False]],
                            col_buy: [['signal', (-1,), ('r.',), False]]},
                        **kwargs_plot)
            plot_series(df, {col_price: '.-k'},
                        cols_styl_up_right={'base_sel': ('--b', False),
                                            col_sel: '.-b'
                                            },
                        cols_to_label_info={
                            col_price: [['signal', (1,), ('gv',), False]],
                            col_sel: [['signal', (1,), ('g.',), False]]},
                        **kwargs_plot)
        plot_series(df, {col_price: '.-k'},
                    cols_to_label_info={col_price:
                                  [['signal', (-1, 1), ('r^', 'gv'), False]]},
                    **kwargs_plot)
    return df['signal']


def cross_cum_maxmin_dif(data, col, rlag, thr_rmax2now, thr_rmin2now,
                         col_price='close', buy_sig=1, sel_sig=-1,
                         plot=True, kwargs_plot={}):
    '''依据指标值与滚动最大最小值的差值绝对大小来生成信号'''
    df = data.copy()
    df['rmax'] = df[col].rolling(rlag).max()
    df['rmin'] = df[col].rolling(rlag).min()
    df['rmax2now'] = df['rmax'] - df[col]
    df['rmin2now'] = df[col] - df['rmin']
    df['signal'] = cross2_plot(df, 'rmax2now', thr_rmax2now, 'rmin2now',
                               thr_rmin2now, buy_sig=buy_sig, sel_sig=sel_sig,
                               col_price=col_price, plot=plot, **kwargs_plot)
    return df['signal']

#%%
if __name__ == '__main__':
    import time
    from dramkit import load_csv
    from dramkit.gentools import get_preval_func_cond
    from dramkit.gentools import replace_repeat_iter
    from dramkit.plottools.plot_candle import plot_candle
    from dramkit.fintools.utils_gains import get_yield_curve

    strt_tm = time.time()

    #%%
    # 50ETF日线行情
    fpath = '../test/510050_daily_pre_fq.csv'
    df = load_csv(fpath)
    df.set_index('date', drop=False, inplace=True)
    # df = df.reindex(columns=['high', 'low', 'close'])

    #%%
    df['acpmp'] = ACPMP(df['close'], periods=range(2, 60),
                        mode='sum', std_type='pct', lag_roll=200,
                        std_order='last')
    plot_series(df, {'close': '.-k'},
                cols_styl_up_right={'acpmp': '.-b'})
    plot_series(df[df['date'] >= '2016-01-01'], {'close': '.-k'},
                cols_styl_up_right={'acpmp': '.-b'})

    #%%
    # CCI
    n = 14
    r = 0.015
    df['cci'] = CCI(df, n=n, r=r)

    df['cci_up_crs'] = cross(df['cci'], 100)
    df['cci_low_crs'] = cross(df['cci'], -100)
    df['signal'] = df[['cci_up_crs', 'cci_low_crs']].apply(lambda x:
                    1 if x['cci_up_crs'] == 1 else \
                    (-1 if x['cci_low_crs'] == -1 else 0), axis=1)
    # # 纠错
    # df['sig_pre'] = get_preval_func_cond(df, 'signal', 'signal', lambda x: x != 0)
    # df['sig_corrt'] = df[['sig_pre', 'cci']].apply(lambda x:
    #             -1 if x['cci'] > 100 and x['sig_pre'] == 1 else \
    #             (1 if x['cci'] < -100 and x['sig_pre'] == -1 else 0), axis=1)
    # df['signal'] = df['signal'] + df['sig_corrt']
    # df['signal'] = replace_repeat_pd(df['signal'], 1, 0)
    # df['signal'] = replace_repeat_pd(df['signal'], -1, 0)
    # df['signal'] = df['signal'].apply(lambda x: -1 if x < 0 else \
    #                                   (1 if x > 0 else 0))

    plot_series(df.iloc[-200:, :], {'close': ('.-k', False)},
                cols_styl_low_left={'cci': ('.-b', False)},
                cols_to_label_info={
                    'cci': [['cci_up_crs', (1,), ('gv',), False],
                            ['cci_low_crs', (-1,), ('r^',), False],],
                    'close': [['signal', (-1, 1), ('r^', 'gv'), False]]},
                xparls_info={'cci': [(100, 'r', '-', 1.3),
                                      (-100, 'r', '-', 1.3)]},
                figsize=(12, 7), grids=True)

    # trade_gain_info, df_gain = get_yield_curve(
    #                             df.iloc[:, :], 'signal',
    #                             base_money=None, base_vol=100,
    #                             fee=1.5/1000, max_loss=None,
    #                             max_gain=None, max_down=None,
    #                             show_dy_maxdown=False,
    #                             func_vol_sub='hold_base_1',
    #                             kwargs_plot={'figsize': (12, 7)})

    #%%
    # KDJ
    df_kdj = KDJ(df, 9, 9, 3)
    df_ = pd.merge(df, df_kdj, how='left', left_index=True, right_index=True)
    plot_series(df_, {'close': '.-k'},
                cols_styl_up_right={'J': '-b'})

    #%%
    # EXPMA
    n = 5
    df['expma'+str(n)] = EXPMA(df['close'], n)

    #%%
    # MACD
    macds = MACD(df['close'])
    df = df.merge(macds, how='left', left_index=True, right_index=True)

    #%%
    # 九转序列
    df_ = df.iloc[:, :].copy()
    n, lag = 9, 4
    df_['dmktd'] = DeMark_TD(df_['close'], n, lag)
    # df_['dmktd'] = replace_repeat_pd(df_['dmktd'], 1, 0)
    # df_['dmktd'] = replace_repeat_pd(df_['dmktd'], -1, 0)
    plot_series(df_, {'close': ('.-k', False)},
                cols_to_label_info={'close':
                    [['dmktd', (-1, 1), ('r^', 'bv'), False]]},
                figsize=(11, 7), grids=True)
    # trade_gain_info, df_gain = get_yield_curve(
    #                             df_, 'dmktd',
    #                             base_money=None, base_vol=100,
    #                             fee=1.5/1000, max_loss=None,
    #                             max_gain=None, max_down=None,
    #                             func_vol_sub='base_1')

    #%%
    # 平均K线
    df_m = mean_candle(df)
    df_m = pd.merge(df.rename(columns={'date': 'time'})[['time']], df_m,
                    how='left', left_index=True, right_index=True)
    plot_candle(df_m.iloc[-200:, :], plot_below=None, figsize=(11, 7))
    df_norm = df.rename(columns={'date': 'time'})
    plot_candle(df_norm.iloc[-200:, :], plot_below=None, figsize=(11, 7))

    #%%
    # RSI
    df['rsi'] = talib_RSI(df['close'], n=24)
    plot_series(df.iloc[-1000:, :], {'close': '.-k'},
                cols_styl_up_right={'rsi': '.-b'})

    #%%
    # BBW
    n = 1500
    df['bbw'] = BBW(df['close'], sign=True)
    # df['bbw'] = df['bbw'].rolling(5).mean()
    df['bbw_dif'] = df['bbw'].diff(1)
    plot_series(df.iloc[-n:, :], {'close': '.-k'},
                cols_styl_up_right={'bbw': '.-b'},
                xparls_info={'bbw': [(0, 'b', '-', 1.0),
                                     (0.06, 'b', '-', 1.0),
                                     (-0.06, 'b', '-', 1.0)]})
    # # df['sig_bbw'] = df['bbw_dif'].apply(lambda x: 1 if x < 0 else \
    # #                                     (-1 if x > 0 else 0))
    # df['sig_bbw'] = df[['bbw', 'bbw_dif']].apply(lambda x:
    #                     1 if x['bbw'] > 0.06 and x['bbw_dif'] < 0 else \
    #                     (-1 if x['bbw'] < -0.06 and x['bbw_dif'] > 0 else 0),
    #                     axis=1)
    # df['sig_bbw'] = replace_repeat_iter(df['sig_bbw'], 1, 0, gap=20)
    # df['sig_bbw'] = replace_repeat_iter(df['sig_bbw'], -1, 0, gap=20)
    # plot_series(df.iloc[-n:, :], {'close': ('.-k', False)},
    #             cols_to_label_info={'close':
    #                 [['sig_bbw', (-1, 1), ('r^', 'bv'), False]]},
    #             figsize=(11, 7), grids=True)
    # trade_gain_info, df_gain = get_yield_curve(
    #                             df.iloc[-n:, :], 'sig_bbw',
    #                             base_money=None, base_vol=100,
    #                             fee=1.5/1000, max_loss=None,
    #                             max_gain=None, max_down=None,
    #                             func_vol_sub='base_1')

    #%%
    # VRI
    df['vri'] = VRI(df, n=5)
    plot_series(df.iloc[-1000:, :], {'close': '.-k'},
                cols_styl_up_right={'vri': '.-b'},
                xparls_info={'vri': [(0, 'b', '-', 1.0)]})

    #%%
    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
