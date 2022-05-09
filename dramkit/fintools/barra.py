# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as lr

from dramkit.logtools.utils_logger import logger_show

#%%
def get_exp_decay_weight(n, adjust=True, logger=None, **kwargs):
    '''
    计算指数衰减权重序列
    
    | x为原始序列，y为加权序列
    | 记y[t] = sum(w[i] * x[i])，权重序列w有多种计算方式
    | 指数加权方式一般为：
    |     y[0) = x[0]
    |     y[t] = (1-a)*y[t-1] + a*x[t]
    |     递推式为：
    |     y[t] = (1-a)^t*x[0] + (1-a)^(t-1)*a*x[1] +···+ (1-a)*a*x[t-1] + a*x[t]

    Parameters
    ----------
    n : int
        权重序列长度
    adjust : bool
        若为False，则采用精确方法计算权重，True采用近似方法
    kwargs : 
        | 设置计算最新一期权重a的参数：
        | - 若kwargs中有参数'a'或'alpha'，则直接将其值作为a；
        | - 若kwargs中有参数'halflife'或'span'或'com'，则采用
            pd.DataFrame.ewm中对应参数的方法计算a；
        | 注：若上述kwargs中参数有多个同时设置，则按
          ['a', 'alpha', 'halflife', 'span', 'com']顺序取第一个。

    Returns
    -------
    w : np.array
        权重序列，返回中w[-1]为最近一期权重，w[0]为最远一期权重

    References
    ----------
    - https://zhuanlan.zhihu.com/p/31412967
    - https://www.zhihu.com/question/49694102
    - https://zh.wikipedia.org/wiki/移動平均
    '''

    def get_a_halflife(halflife):
        '''
        | 半衰期方法计算最新一期权重
        | 
        | 按半衰期，有两种方法确定权重：
        | 方式1：
        |     按照递推公式，精确权重序列为:
        |     w = [(1-a)^t, (1-a)^(t-1)*a, (1-a)^(t-2)*a, ..., (1-a)*a, a]
        |     满足sum(w) = 1
        |     最新一期权重为a，半衰期halflife(h)处权重为(1-a)^h*a，则：
        |     (1-a)^h*a = a/2 ——> a = 1-exp(ln(0.5)/h)
        | 方式2：
        |     记y[t] = a^(t+1)*x[t] + a^t*x[t-1] + ··· + a^2*x[1] + a*x[0]
        |     最新一期权重为a，半衰期halflife(h)处权重为a^(h+1)，则：
        |     a^(h+1) = a/2 ——> a = exp(ln(0.5)/h)
        |     这时得出来的权重序列和并不为1，需要进行标准化w = w/sum(w)
        |     注：为了在最新一期的权重写法跟第一期统一，将a换成1-a
        '''
        assert halflife > 0
        # a = 1 - np.exp(np.log(0.5) / halflife)
        a = 1 - 0.5 ** (1/halflife)
        return a

    def get_a_span(span):
        assert span >= 1
        a = 2 / (span + 1)
        return a

    def get_a_com(com):
        assert com >= 0
        a = 1 / (com + 1)
        return a

    def get_a(**kwargs):
        '''计算a'''
        if 'a' in kwargs:
            return float(kwargs['a'])
        elif 'alpha' in kwargs:
            return float(kwargs['alpha'])
        elif 'halflife' in kwargs:
            return get_a_halflife(**kwargs)
        elif 'span' in kwargs:
            return get_a_span(**kwargs)
        elif 'com' in kwargs:
            return get_a_com(**kwargs)
        else:
            logger_show('最新一期权重`a`计算参数设置有误！', logger, 'err')

    def get_w(a, n):
        '''
        计算权重序列，返回中w[-1]为最近一期权重，w[0]为最远一期权重
        '''
        if adjust:
            w = np.array([(1-a) ** (n-k) for k in range(n)])
            w = w / w.sum()
        else:
            w = np.zeros(n)
            for k in range(n):
                if k == 0:
                    w[k] = (1-a) ** (n-1)
                elif k == n-1:
                    w[k] = a
                else:
                    w[k] = (1-a) ** (n-k-1) * a
        return w

    a = get_a(**kwargs)
    w = get_w(a, n)

    return w


def get_regression_beta(vals_y, vals_x, weights=None, intercept=True,
                        gls_reg=False, algo='ori'):
    '''
    计算beta：一元回归中的斜率项

    Parameters
    ----------
    vals_y : list, np.ndarray, pd.Series
        因变量数据，一维
    vals_x : list, np.ndarray, pd.Series
        自变量数据，一维
    weights : np.array
        一维权重向量。若为None则进行普通回归；若为一维权重向量，则进行加权回归
    intercept : bool
        设置回归是否加入截距项
    gls_reg : bool
        是否用GLS回归（weights不为None时不适用）
    algo : str
        | 计算回归系数算法设置：
        | - 'ori': 用原始样本数据直接计算
        | - 'sklearn': 用sklearn中的线性回归算法计算
        | - 'sm': 用statsmodels.api计算
        | （默认用'ori'速度快些）

    Return
    ------
    beta : float
        beta值，即一元回归系数

    References
    ----------
    - https://max.book118.com/html/2019/0801/8143045042002040.shtm
    - https://zhuanlan.zhihu.com/p/31412967
    '''

    algos = ['ori', 'sklearn', 'sm']
    if not algo in algos:
        raise Exception('算法设置参数`algo`只能为%s！' % algos)

    # 数据转化转化为pd.ndarray
    if not isinstance(vals_y, np.ndarray):
        vals_y = np.array(vals_y)
    if not isinstance(vals_x, np.ndarray):
        vals_x = np.array(vals_x)

    # beta计算
    X = vals_x.reshape(-1, 1)
    X1 = sm.add_constant(X)
    if weights is None:
        if intercept:
            if gls_reg:
                mdl = sm.GLS(vals_y, X1).fit()
                beta = mdl.params[1]
            else:
                if algo == 'ori':
                    # 用样本公式1
                    beta = np.cov(vals_y, vals_x)[0][1] / np.var(vals_x, ddof=1)
                    # 用样本公式2
                    # xmean, ymean = np.mean(vals_x), np.mean(vals_y)
                    # beta = np.sum((vals_x-xmean) * (vals_y-ymean)) / np.sum((vals_x-xmean) ** 2)
                elif algo == 'sklearn':
                    mdl = lr().fit(X, vals_y)
                    beta = mdl.coef_[0]
                elif algo == 'sm':
                    mdl = sm.OLS(vals_y, X1).fit()
                    beta = mdl.params[1]
        else:
            if gls_reg:
                mdl = sm.GLS(vals_y, X).fit()
                beta = mdl.params[0]
            else:
                if algo == 'ori':
                    beta = np.sum(vals_x * vals_y) / np.sum(vals_x ** 2)
                elif algo == 'sklearn':
                    mdl = lr(fit_intercept=False).fit(X, vals_y)
                    beta = mdl.coef_[0]
                elif algo == 'sm':
                    mdl = sm.OLS(vals_y, X).fit()
                    beta = mdl.params[0]
    else:
        if intercept:
            if algo == 'ori':
                xmean = np.sum(weights * vals_x) / np.sum(weights)
                ymean = np.sum(weights * vals_y) / np.sum(weights)
                beta = np.sum(weights * (vals_x-xmean) * (vals_y-ymean)) / np.sum(weights * (vals_x-xmean) ** 2)
            else:
                mdl = sm.WLS(vals_y, X1, weights).fit()
                beta = mdl.params[1]
        else:
            if algo == 'ori':
                beta = np.sum(weights * vals_x * vals_y) / np.sum(weights * vals_x ** 2)
            else:
                mdl = sm.WLS(vals_y, X, weights).fit()
                beta = mdl.params[0]

    return beta


def size(data):
    '''市值因子：流通市值的自然对数'''
    raise NotImplementedError


def get_barra_beta(data, col_pct, col_pct_mkt, rf=3/100, halflife=63,
                   d1year=252, intercept=True, gls_reg=False, algo='ori'):
    '''
    beta因子计算

    Parameters
    ----------
    data : pd.DataFrame
        须包含col_pct和col_pct_mkt指定的两列
    col_pct : str
        col_pct指定列为个股收益率序列数据
    col_pct_mkt : str
        col_pct_mkt指定列为指数收益率数据
    rf : float
        无风险收益（年化）
    halflife : int
        半衰期
    d1year : int
        一年交易天数
    intercept : bool
        回归是否考虑截距
    gls_reg : bool
        是否使用GLS回归
    algo : str
        回归算法选择，参见 :func:`get_regression_beta` 函数
    '''

    df = data.reindex(columns=[col_pct, col_pct_mkt])
    rf = pow(1+rf, 1/d1year) - 1 # 每日无风险收益收益率
    df['expct'] = df['pct'] - rf # 超额收益率
    df['expct_mkt'] = df['pct_mkt'] - rf # 市场超额收益率

    ori_idx = df.index # 保留原来的index
    df.reset_index(drop=True, inplace=True)

    # 滚动计算beta
    df['beta'] = np.nan
    weights = get_exp_decay_weight(d1year, halflife=halflife)
    for k in range(df.shape[0]-d1year+1):
        vals_x = df['expct_mkt'].iloc[k: k+d1year]
        vals_y = df['expct'].iloc[k: k+d1year]
        beta = get_regression_beta(vals_y, vals_x, weights=weights,
                                   intercept=intercept, gls_reg=gls_reg,
                                   algo=algo)
        df.loc[k+d1year-1, 'beta'] = beta

    df.index = ori_idx # 还原index

    return df['beta']

#%%
if __name__ == '__main__':
    strt_tm = time.time()

    #%%
    # 指数衰减权重测试
    a = 0.4
    halflife = 63
    span = 3
    com = 2

    w0 = get_exp_decay_weight(252, alpha=a)
    w1 = get_exp_decay_weight(252, adjust=False, alpha=a)
    w2 = get_exp_decay_weight(252, halflife=halflife)

    np.random.seed(5262)
    df = pd.DataFrame({'x': np.random.randint(1, 10, size=365)})
    # 指数加权移动平均（半衰期权重）
    df['xewm'] = 0
    for k in range(df.shape[0]):
        w = get_exp_decay_weight(n=k+1, adjust=False, alpha=a)
        df.loc[k, 'xewm'] = np.sum(w * df['x'].iloc[:k+1].values)
    # pandas自带指数加权移动平均（半衰期权重），精确权重
    df['xewm0'] = pd.DataFrame.ewm(df['x'], alpha=a,
                                   adjust=False).mean()
    # pandas自带指数加权移动平均（半衰期权重），近似权重
    df['xewm1'] = pd.DataFrame.ewm(df['x'], alpha=a).mean()

    #%%
    # beta测试
    df = pd.DataFrame(columns=['date', 'pct', 'pct_mkt'])
    df['date'] = pd.date_range(start='2018-01-01', end='2018-12-31')
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # 个股每日收益率
    np.random.seed(5262)
    df['pct'] = np.random.uniform(-0.02, 0.02, size=(df.shape[0],))
    # 指数每日收益率
    np.random.seed(62520)
    df['pct_mkt'] = np.random.uniform(-0.02, 0.02, size=(df.shape[0],))

    df['beta'] = get_barra_beta(df, 'pct', 'pct_mkt')
    df['idx'] = range(1, df.shape[0]+1)

    #%%
    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
