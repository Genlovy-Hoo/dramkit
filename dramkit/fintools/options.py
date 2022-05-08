# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm
from math import log, sqrt, exp
from dramkit.plottools.plot_common import plot_series


# 看涨/看跌简称列表
CALL_NAMES = ['c', 'call', '看涨', '看多', '多', '做多', '认购']
PUT_NAMES = ['p', 'put', '看跌', '看空', '空', '做空', '认沽']


def bs_opt(price_now, price_exe, days_left, r=3/100,
           sigma=22.5/100, i_d=0, days1year=365):
    '''
    BS期权定价公式

    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    r : float
        无风险利率，年化
    sigma : float
        波动率（目标标的收益波动率，年化）？
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）

    Returns
    -------
    price_call : float
        认购做多期权合约价格
    price_put : float
        认沽做空期权合约价格

    References
    ----------
    - https://www.optionseducation.org/toolsoptionquotes/optionscalculator
    - http://www.rmmsoft.com.cn/RSPages/onlinetools/OptionAnalysis/OptionAnalysisCN.aspx
    - https://blog.csdn.net/qq_41239584/article/details/83383780
    - https://zhuanlan.zhihu.com/p/38293827
    - https://zhuanlan.zhihu.com/p/38294971
    - https://zhuanlan.zhihu.com/p/96431951
    - https://zhuanlan.zhihu.com/p/142685333（隐含分红率）
    - https://github.com/caly5144/shu-s-project/tree/master/options
    '''

    T = days_left / days1year

    d1 = (log(price_now / price_exe) + \
          (r - i_d + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    # d2 = (log(price_now / price_exe) + \
    #       (r - i_d - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # 看涨
    price_call = price_now * exp(-i_d * T) * norm.cdf(d1) - \
                 price_exe * exp(-r * T) * norm.cdf(d2)
    # 看跌
    price_put = price_exe * exp(-r * T) * norm.cdf(-d2) - \
                price_now * exp(-i_d * T) * norm.cdf(-d1)

    return price_call, price_put


def mc_bs_opt(price_now, price_exe, days_left, r=3/100,
              sigma=22.5/100, days1year=365, mc_cut=60,
              n_mc=500000, random_seed=62, kwargs_plot=None):
    '''
    | 注：没考虑分红！
    | MC-BS，蒙特卡罗模拟BS公式计算期权价格
    
    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    r : float
        无风险利率，年化
    sigma : float
        波动率（目标标的收益波动率，年化）
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）
    mc_cut : int
        将剩余到期时间划分为mc_cut个小时段
    n_mc : int
        蒙特卡罗模拟次数
    random_seed : int, None
        随机数种子

    Returns
    -------
    price_call : float
        认购做多期权合约价格
    price_put : float
        认沽做空期权合约价格    

    References
    ----------
    - https://blog.csdn.net/qwop446/article/details/88914401
    - https://blog.csdn.net/hzk427/article/details/104538974
    '''

    np.random.seed(random_seed)
    
    T = days_left / days1year
    dt = T / mc_cut

    P = np.zeros((mc_cut+1, n_mc)) # 模拟价格
    P[0] = price_now
    for t in range(1, mc_cut+1):
        db = np.random.standard_normal(n_mc) # 布朗运动随机游走
        P[t] = P[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + \
                               sigma * sqrt(dt) * db)

    if not kwargs_plot is None:
        df_plot = pd.DataFrame(P)
        plot_series(df_plot,
                    {x: ('-', False) for x in df_plot.columns},
                    **kwargs_plot)

    # 看涨
    price_call = exp(-r * T) * \
                 np.sum(np.maximum(P[-1] - price_exe, 0)) / n_mc
    # 看跌
    price_put = exp(-r * T) * \
                np.sum(np.maximum(price_exe - P[-1], 0)) / n_mc

    return price_call, price_put


def mc_log_bs_opt(price_now, price_exe, days_left, r=3/100,
                  sigma=22.5/100, days1year=365, mc_cut=60,
                  n_mc=500000, random_seed=62, kwargs_plot=None):
    '''
    | 注：没考虑分红！
    | MC-BS，蒙特卡罗模拟BS公式计算期权价格，对数格式

    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    r : float
        无风险利率，年化
    sigma : float
        波动率（目标标的收益波动率，年化）
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）
    mc_cut : int
        将剩余到期时间划分为mc_cut个小时段
    n_mc : int
        蒙特卡罗模拟次数
    random_seed : int, None
        随机数种子
    
    Returns
    -------
    price_call : float
        认购做多期权合约价格
    price_put : float
        认沽做空期权合约价格    

    References
    ----------
    - https://blog.csdn.net/qwop446/article/details/88914401
    - https://blog.csdn.net/hzk427/article/details/104538974
    '''

    np.random.seed(random_seed)

    T = days_left / days1year
    dt = T / mc_cut

    # 布朗运动随机游走
    dbs = np.random.standard_normal((mc_cut+1, n_mc))
    dbs = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * \
                    sqrt(dt) * dbs, axis=0)
    P = price_now * np.exp(dbs) # 模拟价格

    if not kwargs_plot is None:
        df_plot = pd.DataFrame(P)
        plot_series(df_plot,
                    {x: ('-', False) for x in df_plot.columns},
                    **kwargs_plot)

    # 看涨
    price_call = exp(-r * T) * \
                 np.sum(np.maximum(P[-1] - price_exe, 0)) / n_mc
    # 看跌
    price_put = exp(-r * T) * \
                np.sum(np.maximum(price_exe - P[-1], 0)) / n_mc

    return price_call, price_put


def bopm_european(price_now, price_exe, days_left, r=3/100,
                  sigma=22.5/100, days1year=365, mc_cut=60):
    '''
    | 注：没考虑分红！
    | 二叉树欧式期权定价模型（BOP/CRR）

    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    r : float
        无风险利率，年化
    sigma : float
        波动率（目标标的收益波动率，年化）
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）
    mc_cut : int
        将剩余到期时间划分为mc_cut个小时段

    Returns
    -------
    price_call : float
        认购做多期权合约价格
    price_put : float
        认沽做空期权合约价格 

    References
    ----------
    - https://zhuanlan.zhihu.com/p/62031783
    - https://wiki.mbalib.com/wiki/二项期权定价模型
    '''
    
    T = days_left / days1year
    dt = T / mc_cut
    df = exp(-r * dt) # discount per interval？

    # 计算u，d，p
    u = exp(sigma * sqrt(dt)) # up movement
    d = 1 / u # down movement
    p = (exp(r * dt) - d) / (u - d) # martingale branch probability

    # 二叉树
    mu = np.arange(mc_cut + 1)
    mu = np.resize(mu, (mc_cut + 1, mc_cut + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md

    # 计算各节点的股票价格
    P = price_now * mu * md

    # 计算叶子结点的期权价值
    Vcal = np.maximum(P-price_exe, 0) # 看涨期权
    Vput = np.maximum(price_exe-P, 0) # 看跌期权

    # 逐步向前加权平均并折现，得到期初期权价值
    for k in range(0, mc_cut):
        # 逐列更新期权价值，相当于二叉树中的逐层向前折算
        Vcal[0:mc_cut-k, mc_cut-k-1] = \
                (p * Vcal[0:mc_cut - k, mc_cut - k] + \
                 (1 - p) * Vcal[1:mc_cut - k + 1, mc_cut - k]) * df
        Vput[0:mc_cut-k, mc_cut-k-1] = \
                (p * Vput[0:mc_cut - k, mc_cut - k] + \
                 (1 - p) * Vput[1:mc_cut - k + 1, mc_cut - k]) * df

    price_call, price_put = Vcal[0, 0], Vput[0, 0]
    return price_call, price_put


# # CRR美式期权
# def bopm_american(price_now, K, T, r, sigma, otype, M=4):
#     '''
#     注：没考虑分红！
#     二叉树美式期权定价模型（BOP/CRR）

#     Parameters
#     ----------
#     price_now : float
#         stock/index level at time 0
#     K : float
#         strike price
#     T : float
#         date of maturity
#     r : float
#         constant, risk-less short rate
#     sigma : float
#         volatility
#     otype : string
#         either 'call' or 'put'
#     M : int
#         number of time intervals

#     Returns
#     -------
#     price_call: 认购做多期权合约价格
#     price_put: 认沽做空期权合约价格

#     参考：
#         https://zhuanlan.zhihu.com/p/62031783
#         https://wiki.mbalib.com/wiki/二项期权定价模型
#     '''
#     # 一.生成二叉树
#     dt = T / M  # length of time interval
#     df = exp(-r * dt)  # discount per interval
#     inf = exp(r * dt)  # discount per interval

#     # 计算udp
#     u = exp(sigma * sqrt(dt))  # up movement
#     d = 1 / u  # down movement
#     q = (exp(r * dt) - d) / (u - d)  # martingale branch probability

#     # 初始化幂矩阵
#     mu = np.arange(M + 1)
#     mu = np.resize(mu, (M + 1, M + 1))
#     md = np.transpose(mu)

#     # 计算个节点单向变动时的股票价格
#     mus = u ** (mu - md)
#     mds = d ** md

#     # 得到各节点的股票价格
#     S = price_now * mus * mds

#     # 二.计算每个节点股票的预期价格
#     mes = price_now * inf ** mu

#     # 三.得到叶子结点的期权价值
#     if otype == 'call':
#         V = np.maximum(S - K, 0)
#         #计算每个节点提前行权的收益
#         oreturn = mes - K
#     else:
#         V = np.maximum(K - S, 0)
#         #计算每个节点提前行权的收益
#         oreturn = K - mes

#     # 四.逐步向前加权平均折现和提前行权的收益比较，得到期初期权价值
#     for z in range(0, M):  # backwards iteration
#         #计算后期折现的后期价格
#         ovalue = (q * V[0:M - z, M - z] +
#                           (1 - q) * V[1:M - z + 1, M - z]) * df
#         #逐列更新期权价值，相当于二叉树中的逐层向前折算
#         #期权价格取后期折现和提前行权获得收益的最大值
#         V[0:M - z, M - z - 1] = np.maximum(ovalue, oreturn[0:M - z, M - z - 1])
#         # 原文评论区纠错代码
          # V[0:M - z, M - z - 1] = np.maximum(ovalue, V[0:M - z, M - z - 1])

#     return V[0, 0]


def bsm_iv_dichotomy(price_now, price_exe, price_opt, days_left,
                     opt_type, r=3/100, i_d=0.0, days1year=365,
                     sigma_max=3.0, sigma_min=0.0, tol=1e-6,
                     n_tol=100, max_iter=10000):
    '''
    二分法求隐含波动率

    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    price_opt : float
        期权现价
    days_left : int
        剩余天数
    opt_type : str
        期权类型
    r : float
        无风险利率，年化
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）
    sigma_max : float
        隐含波动率上限
    sigma_min : float
        隐含波动率下限
    tol : float
        迭代误差控制
    n_tol : int
        当计算的理论期权价格连续n_tol次迭代改变量均不超过tol时结束
    max_iter : int
        迭代最大次数控制

    Returns
    -------
    sigma_iv : float
        返回隐含波动率求解结果

    References
    ----------
    - https://zhuanlan.zhihu.com/p/142685333
    - https://www.jianshu.com/p/e73f538859df
    '''

    sigma_top, sigma_floor = sigma_max, sigma_min
    sigma_iv = (sigma_floor + sigma_top) / 2 # 隐含波动率二分法初始值
    price_opt_est, price_opt_last = 0, 0 # 期权价格估计值初始化
    Cnt = 0 # 计数器
    dif, ndif, last_tol_OK = np.inf, 0, False
    while dif > tol or ndif < n_tol:
        if opt_type.lower() in CALL_NAMES:
            price_opt_est, _ = bs_opt(price_now, price_exe,
                                      days_left, r=r,
                                      sigma=sigma_iv, i_d=i_d,
                                      days1year=days1year)
        elif opt_type.lower() in PUT_NAMES:
            _, price_opt_est = bs_opt(price_now, price_exe,
                                      days_left, r=r,
                                      sigma=sigma_iv, i_d=i_d,
                                      days1year=days1year)

        dif = abs(price_opt_last - price_opt_est)
        if dif <= tol and last_tol_OK:
            ndif += 1
        if dif <= tol:
            last_tol_OK = True
        else:
            last_tol_OK = False
            ndif = 0

        price_opt_last = price_opt_est

        # 根据价格判断波动率是被低估还是高估，并对隐含波动率进行修正
        if price_opt - price_opt_est > 0:
            sigma_floor = sigma_iv
            sigma_iv = (sigma_iv + sigma_top) / 2
        elif price_opt - price_opt_est < 0:
            sigma_top = sigma_iv
            sigma_iv = (sigma_iv + sigma_floor) / 2
        else:
            return sigma_iv

        # 注：时间价值为0的期权是算不出隐含波动率的，因此设置检查机制，
        # 迭代到一定次数就不再继续了
        Cnt += 1
        if Cnt > max_iter:
            # sigma_iv = 0
            break

    return sigma_iv


def bsm_iv_dichotomy0(price_now, price_exe, price_opt, days_left,
                      opt_type, r=3/100, i_d=0.0, days1year=365,
                      sigma_max=3.0, sigma_min=0.0, tol=1e-6,
                      max_iter=10000):
    '''
    二分法求隐含波动率
    
    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    price_opt : float
        期权现价
    days_left : int
        剩余天数
    opt_type : str
        期权类型
    r : float
        无风险利率，年化
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）
    sigma_max : float
        隐含波动率上限
    sigma_min : float
        隐含波动率下限
    tol : float
        迭代误差控制
    max_iter : int
        迭代最大次数控制
    
    Returns
    -------
    sigma_iv : float
        返回隐含波动率求解结果

    References
    ----------
    - https://zhuanlan.zhihu.com/p/142685333
    - https://www.jianshu.com/p/e73f538859df
    '''

    sigma_top, sigma_floor = sigma_max, sigma_min
    sigma_iv = (sigma_floor + sigma_top) / 2 # 隐含波动率二分法初始值
    price_opt_est = 0 # 期权价格估计值初始化
    Cnt = 0 # 计数器
    while abs(price_opt - price_opt_est) > tol:
        if opt_type.lower() in CALL_NAMES:
            price_opt_est, _ = bs_opt(price_now, price_exe,
                                      days_left, r=r,
                                      sigma=sigma_iv, i_d=i_d,
                                      days1year=days1year)
        elif opt_type.lower() in PUT_NAMES:
            _, price_opt_est = bs_opt(price_now, price_exe,
                                      days_left, r=r,
                                      sigma=sigma_iv, i_d=i_d,
                                      days1year=days1year)

        # 根据价格判断波动率是被低估还是高估，并对隐含波动率进行修正
        if price_opt - price_opt_est > 0:
            sigma_floor = sigma_iv
            sigma_iv = (sigma_iv + sigma_top) / 2
        elif price_opt - price_opt_est < 0:
            sigma_top = sigma_iv
            sigma_iv = (sigma_iv + sigma_floor) / 2
        else:
            return sigma_iv

        # 注：时间价值为0的期权是算不出隐含波动率的，因此设置检查机制，
        # 迭代到一定次数就不再继续了
        Cnt += 1
        if Cnt > max_iter:
            return sigma_iv
            # sigma_iv = 0
            # break

    return sigma_iv


def bsm_iv_newton(price_now, price_exe, price_opt, days_left,
                  opt_type, r=3/100, i_d=0.0, sigma_iv_init=None,
                  days1year=365, lr=0.1, max_iter=1000):
    '''
    牛顿法计算隐含波动率

    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    price_opt : float
        期权现价
    days_left : int
        剩余天数
    opt_type : str
        期权类型
    r : float
        无风险利率，年化
    i_d : float
        隐含分红率？
    sigma_iv_init : float
        设置隐含波动率初始值
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）
    lr : float
        学习率
    max_iter : int
        迭代最大次数控制

    Returns
    -------
    sigma_iv : float
        返回隐含波动率求解结果

    References
    ----------
    https://www.jianshu.com/p/e73f538859df
    '''

    if sigma_iv_init is None:
        sigma_iv_init = 1
    sigma_iv = sigma_iv_init

    k = 0
    while k < max_iter:
        if opt_type.lower() in CALL_NAMES:
            price_opt_est, _ = bs_opt(price_now, price_exe,
                                      days_left, r=r,
                                      sigma=sigma_iv, i_d=i_d,
                                      days1year=days1year)
        elif opt_type.lower() in PUT_NAMES:
            _, price_opt_est = bs_opt(price_now, price_exe,
                                      days_left, r=r,
                                      sigma=sigma_iv, i_d=i_d,
                                      days1year=days1year)

        vega = bs_vega(price_now, price_exe, days_left, r=r,
                       sigma=sigma_iv, i_d=i_d, days1year=days1year)

        sigma_iv -= lr * (price_opt_est - price_opt) / vega

        k += 1

    return sigma_iv


def bs_vega(price_now, price_exe, days_left,
            r=3/100, sigma=25/100, i_d=0, days1year=365):
    '''
    BS公式，Vega计算

    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    r : float
        无风险利率，年化
    sigma : float
        波动率
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）

    Returns
    -------
    vega : float
        返回vega求解结果

    References
    ----------
    - https://blog.csdn.net/zita_11/article/details/104200887
    - https://zhuanlan.zhihu.com/p/135867175
    '''

    T = days_left / days1year
    d1 = (log(price_now / price_exe) + \
          (r - i_d + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    vega = price_now * exp(-i_d * T) * norm.pdf(d1) * sqrt(T)

    return vega


def bs_delta(price_now, price_exe, days_left, opt_type,
             r=3/100, sigma=25/100, i_d=0.0, days1year=365):
    '''
    | 注：分红处理需要再检查确认！
    | BS公式，Delta计算
    
    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    opt_type : str
        期权类型
    r : float
        无风险利率，年化
    sigma : float
        波动率
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）

    Returns
    -------
    delta : float
        返回delta求解结果
        
    References
    ----------
    - https://blog.csdn.net/zita_11/article/details/104200887
    - https://zhuanlan.zhihu.com/p/113915774
    '''

    T = days_left / days1year
    d1 = (log(price_now / price_exe) + \
          (r - i_d + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    if opt_type.lower() in CALL_NAMES:
        delta = norm.cdf(d1)
    elif opt_type.lower() in PUT_NAMES:
        delta = -norm.cdf(-d1)

    return delta


def bs_gamma(price_now, price_exe, days_left,
             r=3/100, sigma=25/100, i_d=0.0, days1year=365):
    '''
    | 注：分红处理需要再检查确认！
    | BS公式，Gamma计算
    
    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    r : float
        无风险利率，年化
    sigma : float
        波动率
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）

    Returns
    -------
    gamma : float
        返回gamma求解结果

    References
    ----------
    - https://blog.csdn.net/zita_11/article/details/104200887
    - https://zhuanlan.zhihu.com/p/113915774
    '''

    T = days_left / days1year
    d1 = (log(price_now / price_exe) + \
          (r - i_d + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    gamma = norm.pdf(d1) / (price_now * sigma * sqrt(T))

    return gamma


def bs_theta(price_now, price_exe, days_left, opt_type,
             r=3/100, sigma=25/100, i_d=0.0, days1year=365):
    '''
    | 注：分红处理需要再检查确认！
    | BS公式，Theta计算
    
    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    opt_type : str
        期权类型
    r : float
        无风险利率，年化
    sigma : float
        波动率
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）

    Returns
    -------
    theta : float
        返回theta求解结果

    References
    ----------
    - https://blog.csdn.net/zita_11/article/details/104200887
    - https://zhuanlan.zhihu.com/p/113915774
    '''

    T = days_left / days1year
    d1 = (log(price_now / price_exe) + \
          (r - i_d + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    tmp1 = -1 * (price_now * norm.pdf(d1) * sigma) / (2 * sqrt(T))
    if opt_type.lower() in CALL_NAMES:
        n = 1
    elif opt_type.lower() in PUT_NAMES:
        n = -1
    tmp2 = n * r * price_exe * np.exp(-r * T) * norm.cdf(n * d2)

    theta = (tmp1 - tmp2) / days1year

    return theta


def bs_rho(price_now, price_exe, days_left, opt_type,
           r=3/100, sigma=25/100, i_d=0.0, days1year=365):
    '''
    | 注：分红处理需要再检查确认！
    | BS公式，Rho计算
    
    Parameters
    ----------
    price_now : float
        标的现价
    price_exe : float
        标的执行价
    days_left : int
        剩余天数
    opt_type : str
        期权类型
    r : float
        无风险利率，年化
    sigma : float
        波动率
    i_d : float
        隐含分红率？
    days1year : int
        一年天数，默认为365，即自然天数（也可以只考虑交易天数）

    Returns
    -------
    rho : float
        返回rho求解结果
    
    References
    ----------
    https://zhuanlan.zhihu.com/p/137938792
    '''

    T = days_left / days1year
    d1 = (log(price_now / price_exe) + \
          (r - i_d + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if opt_type.lower() in CALL_NAMES:
        rho = T * price_exe * exp(-r * T) * norm.cdf(d2)
    elif opt_type.lower() in PUT_NAMES:
        rho = -1 * T * price_exe * exp(-r * T) * norm.cdf(-d2)

    return rho


if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings('ignore')

    # BS公式&蒙特卡罗测试--------------------------------------------------------
    price_now = 3.731
    price_exe = 3.5
    days_left = 100
    r = 3/100
    i_d = 0.0/100
    sigma = 25/100
    days1year = 365
    n_mc = 1000
    random_seed = None

    # BS|蒙特卡罗|BOPM计算期权理论价格
    Popt_bs = bs_opt(price_now, price_exe, days_left, r=r, sigma=sigma, i_d=i_d, days1year=days1year)
    Popt_mc = mc_bs_opt(price_now, price_exe, days_left, r=r, sigma=sigma, days1year=days1year,
                       n_mc=n_mc, random_seed=random_seed, kwargs_plot=None)
    Popt_mc_log = mc_log_bs_opt(price_now, price_exe, days_left, r=r, sigma=sigma, days1year=days1year,
                              n_mc=n_mc, random_seed=random_seed,
                              kwargs_plot=None)
    Popt_bopm = bopm_european(price_now, price_exe, days_left, r=r, sigma=sigma, days1year=days1year)
    print('看涨：')
    print('Popt_bs: {}'.format(round(Popt_bs[0], 6)))
    print('Popt_mc: {}'.format(round(Popt_mc[0], 6)))
    print('Popt_mclog: {}'.format(round(Popt_mc_log[0], 6)))
    print('Popt_bopm: {}'.format(round(Popt_bopm[0], 6)))
    print('看跌：')
    print('Popt_bs: {}'.format(round(Popt_bs[1], 6)))
    print('Popt_mc: {}'.format(round(Popt_mc[1], 6)))
    print('Popt_mclog: {}'.format(round(Popt_mc_log[1], 6)))
    print('Popt_bopm: {}'.format(round(Popt_bopm[1], 6)))


    # BS公式计算隐含波动率
    # price_opt = 0.0702
    # opt_type = 'p'
    price_opt = 0.1071
    opt_type = 'c'

    sigma_iv = bsm_iv_dichotomy(price_now, price_exe, price_opt, days_left, opt_type, r=r, i_d=i_d)
    print('二分法隐波: {}'.format(round(sigma_iv, 10)))
    sigma_iv = bsm_iv_newton(price_now, price_exe, price_opt, days_left, opt_type, r=r, i_d=i_d)
    print('牛顿法隐波: {}'.format(round(sigma_iv, 10)))


    # BS-vega
    vega = bs_vega(price_now, price_exe, days_left, r=r, sigma=sigma, i_d=i_d, days1year=days1year)
    print('BS-vega: {}'.format(round(vega, 6)))


    # BS-delta
    delta = bs_delta(price_now, price_exe, days_left, opt_type, r=r, sigma=sigma, i_d=i_d,
                     days1year=days1year)
    print('BS-delta: {}'.format(round(delta, 6)))


    # BS-gamma
    gamma = bs_gamma(price_now, price_exe, days_left, r=r, sigma=sigma, i_d=i_d, days1year=days1year)
    print('BS-gamma: {}'.format(round(gamma, 6)))


    # BS-theta
    theta = bs_theta(price_now, price_exe, days_left, opt_type, r=r, sigma=sigma, i_d=i_d,
                     days1year=days1year)
    print('BS-theta: {}'.format(round(theta, 6)))


    # BS-rho
    rho = bs_rho(price_now, price_exe, days_left, opt_type, r=r, sigma=sigma, i_d=i_d,
                 days1year=days1year)
    print('BS-rho: {}'.format(round(rho, 6)))
