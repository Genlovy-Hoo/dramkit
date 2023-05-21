# -*- coding: utf-8 -*-

import time
import math
import random
import numpy as np
from dramkit.gentools import isnull
from dramkit.optimizer.utils_heuristic import rand_init


def _update_best(nests, nests_new, fvals, popsize, dim, objf, **kwargs):
    '''
    | 若nests_new中的个体（样本）优于nests中的样本，则用nests_new更新nests
    | 同时记录新个体|样本中的最优值和最优解
    '''

    # 更新nests
    for k in range(0, popsize):
        val = objf(nests_new[k, :], **kwargs)
        if val < fvals[k]:
            fvals[k] = val
            nests[k, :] = nests_new[k, :]

    # 最优个体
    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    nest_best = nests[best_idx, :]

    return best_val, nest_best, nests, fvals


def _update_levy(nests, gBest, x_lb, x_ub, popsize, dim, beta=1.5, alpha=0.01):
    '''
    | Levy飞行
    | 参考：
    | sigma、u、v、s意义见https://www.jianshu.com/p/4f6e02fc8396
    '''

    tmp1 = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    tmp2 = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (tmp1 / tmp2) ** (1 / beta)

    u = sigma * np.random.randn(popsize, dim) # 正态分布 u ~ N(0, sigma^2)
    v = np.random.randn(popsize, dim) # 标准正态分布 v ~ N(0, 1)

    s = u / (abs(v) ** (1 / beta)) # 步长

    stepsize = alpha * (s * (nests - gBest)) # alpha相当于学习率？
    nests_new = nests + stepsize * np.random.randn(popsize, dim)
    nests_new = np.clip(nests_new, x_lb, x_ub) # 越界处理

    return nests_new


def _replace_nests(nests, pa, popsize, dim, x_lb, x_ub):
    '''
    个体|样本以pa概率进行位置变化
    '''

    # 每个个体|样本身边需要重新生成
    K = np.random.uniform(0, 1, (popsize, dim)) > pa
    # 更新步长
    stepsize = random.random() * (nests[np.random.permutation(popsize), :] - \
                                  nests[np.random.permutation(popsize), :])
    nests_new = nests + stepsize * K # 新个体|样本
    nests_new = np.clip(nests_new, x_lb, x_ub) # 越界处理

    return nests_new


def cs(objf, func_opter_parms):
    '''
    布谷鸟搜索算法(Cuckoo Search) CS algorithm

    TODO
    ----
    目前仅考虑自变量连续实数情况，以后可增加自变量为离散的情况

    Parameters
    ----------
    objf : function
        目标函数。注：须事先转化为求极小值问题
    func_opter_parms : FuncOpterInfo
        :class:`dramkit.optimizer.utils_heuristic.FuncOpterInfo` 类，
        须设置parms_func、parms_opter、parms_log

        | parms_func为目标函数参数信息dict，key须包含:
        |     x_lb: 自变量每个维度取值下界，list或数值，为list时长度应等于dim
        |     x_ub: 自变量每个维度取值上界，list或数值，为list时长度应等于dim
        |     dim: 自变量维度数
        |     kwargs: 目标函数接收的其它参数
        | parms_opter: 优化函数参数信息dict，key须包含:
        |     popsize: 群体数量（每轮迭代的样本数量）
        |     max_iter: 最大迭代寻优次数
        |     pa: 鸟巢被发现概率
        |     alpha, beta: Levy飞行参数
        | parms_log: 日志参数信息dict，key须包含:
        |     logger: 日志记录器
        |     nshow: 若为整数，则每隔nshow轮日志输出当前最优目标函数值

    Returns
    -------
    func_opter_parms : FuncOpterInfo
        更新优化过程之后的func_opter_parms

    References
    ----------
    - https://blog.csdn.net/u013631121/article/details/76944879
    - https://www.jianshu.com/p/4f6e02fc8396
    - https://github.com/7ossam81/EvoloPy
    '''

    # 参数提取
    opter_name = func_opter_parms.parms_opter['opter_name']
    if opter_name == '' or isnull(opter_name):
        opter_name  = 'cs'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    popsize = func_opter_parms.parms_opter['popsize']
    max_iter = func_opter_parms.parms_opter['max_iter']
    pa = func_opter_parms.parms_opter['pa']
    beta = func_opter_parms.parms_opter['beta']
    alpha = func_opter_parms.parms_opter['alpha']
    # 日志参数
    logger = func_opter_parms.parms_log['logger']
    nshow = func_opter_parms.parms_log['nshow']

    # 时间记录
    strt_tm = time.monotonic()
    func_opter_parms.set_start_time(time.strftime('%Y-%m-%d %H:%M:%S'))


    # 边界统一为列表
    if not isinstance(x_lb, list):
        x_lb = [x_lb] * dim
    if not isinstance(x_ub, list):
        x_ub = [x_ub] * dim


    # 初始化
    nests = rand_init(popsize, dim, x_lb, x_ub) # 鸟巢（个体or样本）随机初始化
    nests_new = nests.copy()

    gBest = np.zeros(dim) # 全局最优解
    gBestVal = np.inf # 全局最优值

    fvals = np.zeros(popsize) # 存放每个个体的目标函数值
    fvals.fill(float('inf')) # 最小值问题初始化为正无穷大

    # 保存收敛过程
    convergence_curve = np.zeros(max_iter) # 全局最优值
    convergence_curve_mean = np.zeros(max_iter) # 平均值

    # 初始最优解
    gBestVal, gBest, nests, fvals = _update_best(nests, nests_new, fvals,
                                                popsize, dim, objf, **kwargs)


    # 迭代寻优
    for l in range(0, max_iter):
        # Levy flights
        nests_new = _update_levy(nests, gBest, x_lb, x_ub, popsize, dim,
                                beta=beta, alpha=alpha)

        # 个体|样本更新
        _, _, nests, fvals = _update_best(nests, nests_new, fvals, popsize,
                                         dim, objf, **kwargs)

        # 每个个体|样本以pa概率进行位置变化
        nests_new = _replace_nests(nests_new, pa, popsize, dim, x_lb, x_ub)

        # 个体|样本更新并获取最优个体|样本
        best_val, nest_best, nests, fvals = _update_best(nests, nests_new,
                                            fvals, popsize, dim, objf, **kwargs)

        if best_val < gBestVal:
           gBestVal = best_val
           gBest = nest_best

        # 每轮迭代都保存最优目标值
        convergence_curve[l] = gBestVal
        convergence_curve_mean[l] = np.mean(fvals)

        if nshow:
           if (l+1) % nshow ==0:
               opter_name = func_opter_parms.parms_opter['opter_name']
               func_name = func_opter_parms.parms_func['func_name']
               logger.info('{} for {}, iter: {}, '.format(opter_name, func_name, l+1) + \
                           'best fval: {}'.format(gBestVal))


    # 更新func_opter_parms
    end_tm = time.monotonic()
    func_opter_parms.set_end_time(time.strftime('%Y-%m-%d %H:%M:%S'))
    func_opter_parms.set_exe_time(end_tm-strt_tm)
    func_opter_parms.set_convergence_curve(convergence_curve)
    func_opter_parms.set_convergence_curve_mean(convergence_curve_mean)
    func_opter_parms.set_best_val(gBestVal)
    func_opter_parms.set_best_x(gBest)

    return func_opter_parms


if __name__ == '__main__':
    import pandas as pd
    from dramkit.optimizer.base_funcs import TestFuncs
    from dramkit.optimizer.utils_heuristic import FuncOpterInfo
    from dramkit import plot_series, simple_logger, TimeRecoder
    from dramkit.logtools.logger_general import get_logger
    from dramkit.logtools.utils_logger import close_log_file


    tr = TimeRecoder()

    objf = TestFuncs.ackley2
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -10, 'x_ub': 10, 'dim': 10, 'kwargs': {}}
    parms_opter = {'opter_name': 'cs-test',
                   'popsize': 20, 'max_iter': 1000,
                   'pa': 0.2, 'beta': 1.5, 'alpha': 0.01}
    # logger = simple_logger()
    logger = get_logger('./_test/log/cs_test.txt')
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 100}

    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = cs(objf, func_opter_parms)

    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(10, 6))

    best_x = func_opter_parms.best_x
    func_opter_parms.parms_log['logger'].info('best x: {}'.format(best_x))

    close_log_file(logger)


    tr.used()
