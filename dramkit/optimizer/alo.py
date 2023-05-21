# -*- coding: utf-8 -*-

import time
import random
import numpy as np
from dramkit.gentools import isnull
from dramkit.optimizer.utils_heuristic import rand_init
from dramkit.optimizer.utils_heuristic import sort_population


def _roulette_wheel_selection_id(fvals):
    '''
    轮盘赌法：个体函数值越小（最小值问题），越容易被选中

    Parameters
    ----------
    fvals : np.array
        所有个体值列表

    Returns
    -------
    individual_id : int
        被选中的个体序号
    '''

    # 最小值问题转化
    reverse = max(fvals) + min(fvals)
    reverseScores = reverse - fvals

    sumScores = sum(reverseScores)
    pick = random.uniform(0, sumScores)
    current = 0
    for individual_id in range(len(fvals)):
        current += reverseScores[individual_id]
        if current > pick:
            return individual_id


def _random_walk_around_antlion(antlion, now_iter, Max_iter, x_lb, x_ub, dim):
    '''随机游走'''

    # ALO蚁狮算法.pdf公式(2.10)、(2.11)
    if now_iter >= Max_iter * 0.95:
        I = 1 + 10**6 * (now_iter / Max_iter)
    elif now_iter >= Max_iter * 0.9:
        I = 1 + 10**5 * (now_iter / Max_iter)
    elif now_iter >= Max_iter * 0.75:
        I = 1 + 10**4 * (now_iter / Max_iter)
    elif now_iter >= Max_iter * 0.5:
        I = 1 + 10**3 * (now_iter / Max_iter)
    else:
        I = 1 + 10**2 * (now_iter / Max_iter)
    x_lb_ = np.array(x_lb) / I
    x_ub_ = np.array(x_ub) / I

    # ALO蚁狮算法.pdf公式(2.8)、(2.9)
    if random.random() < 0.5:
        x_lb_ = x_lb_ + antlion
    else:
        x_lb_ = -x_lb_ + antlion
    if random.random() >= 0.5:
        x_ub_ = x_ub_ + antlion
    else:
        x_ub_ = -x_ub_ + antlion

    # 随机游走过程（当Max_iter比较大时这里很慢，需要改进？）
    RWs = np.zeros((Max_iter+1, dim))
    for dm in range(dim):
        # ALO蚁狮算法.pdf公式(2.1)、(2.2)
        X = [0] + [1 if random.random() > 0.5 else -1 for _ in range(Max_iter)]
        X = np.cumsum(X)

        # ALO蚁狮算法.pdf公式(2.7)？
        a, b = min(X), max(X)
        c, d = x_lb_[dm], x_ub_[dm]
        aa = [a for _ in range(Max_iter+1)]
        X_norm = [(x-y) * (d-c) / (b-a) + c for x, y in zip(X, aa)]

        for t in range(len(X_norm)):
            RWs[t][dm] = X_norm[t]

    return RWs


def alo(objf, func_opter_parms):
    '''
    蚁狮优化算法(Ant Lion Optimizer) ALO algorithm

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
        | parms_log: 日志参数信息dict，key须包含:
        |     logger: 日志记录器
        |     nshow: 若为整数，则每隔nshow轮日志输出当前最优目标函数值

    Returns
    -------
    func_opter_parms : FuncOpterInfo
        更新优化过程之后的func_opter_parms

    References
    ----------
    - ALO蚁狮算法.pdf
    - https://github.com/zhaoxingfeng/ALO
    - https://github.com/7ossam81/EvoloPy
    '''

    # 参数提取
    opter_name = func_opter_parms.parms_opter['opter_name']
    if opter_name == '' or isnull(opter_name):
        opter_name  = 'alo'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    popsize = func_opter_parms.parms_opter['popsize']
    max_iter = func_opter_parms.parms_opter['max_iter']
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
    Xantlion = rand_init(popsize, dim, x_lb, x_ub) # 蚁狮位置
    Xant = rand_init(popsize, dim, x_lb, x_ub) # 蚂蚁位置

    gBest = np.zeros(dim) # 全局最优解（精英蚁狮）
    gBestVal = np.inf # 全局最优值

    fvals_antlion = np.zeros(popsize) # 存放每个蚁狮的目标函数值
    fvals_antlion.fill(float('inf'))
    fvals_ant = np.zeros(popsize) # 存放每个蚂蚁的目标函数值
    fvals_ant.fill(float('inf'))

    # 保存收敛过程
    convergence_curve = np.zeros(max_iter) # 全局最优值
    convergence_curve_mean = np.zeros(max_iter) # 平均值


    # 蚁狮排序，最优一个为精英蚁狮
    for i in range(popsize):
        fvals_antlion[i] = objf(Xantlion[i], **kwargs)
    XantlionSorted, FvalsSorted_antlion = sort_population(Xantlion,
                                                         fvals_antlion)
    gBest = XantlionSorted[0]
    gBestVal = FvalsSorted_antlion[0]


    # 迭代寻优
    for l in range(0, max_iter):
        for i in range(popsize):
            RoletteId = _roulette_wheel_selection_id(FvalsSorted_antlion)
            # ALO蚁狮算法.pdf公式(2.13)
            RA = _random_walk_around_antlion(XantlionSorted[RoletteId],
                                        l, max_iter, x_lb, x_ub, dim)
            RE = _random_walk_around_antlion(gBest, l, max_iter, x_lb, x_ub, dim)
            Xant[i] = [(x + y)/2 for x, y in zip(RA[l], RE[l])]
        Xant = np.clip(Xant, x_lb, x_ub)
        for j in range(popsize):
            fvals_ant[j] = objf(Xant[j], **kwargs)

        # 蚂蚁和蚁狮合并
        Xall = np.concatenate((XantlionSorted, Xant), axis=0)
        fvals_all = np.concatenate((FvalsSorted_antlion, fvals_ant), axis=0)
        # 蚁狮位置保持由于蚂蚁位置
        XallSorted, FvalsAllSorted = sort_population(Xall, fvals_all)
        fvals_antlion = FvalsAllSorted[0:popsize]
        XantlionSorted = XallSorted[0:popsize]

        if fvals_antlion[0] <= gBestVal:
            gBestVal = fvals_antlion[0]
            gBest = XantlionSorted[0]

        XantlionSorted[0] = gBest
        fvals_antlion[0] = gBestVal

        # 每轮迭代都保存最优目标值
        convergence_curve[l] = gBestVal
        convergence_curve_mean[l] = np.mean(fvals_antlion)

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

    # objf = TestFuncs.f5
    # parms_func = {'func_name': objf.__name__,
    #               'x_lb': -10, 'x_ub': 10, 'dim': 5, 'kwargs': {}}
    # parms_opter = {'opter_name': 'alo-test',
    #                 'popsize': 20, 'max_iter': 1000}
    objf = TestFuncs.f12
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -1, 'x_ub': 1, 'dim': 4, 'kwargs': {}}
    parms_opter = {'opter_name': 'alo-test',
                   'popsize': 10, 'max_iter': 80}
    # logger = simple_logger()
    logger = get_logger('./_test/log/alo_test.txt')
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 1}

    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = alo(objf, func_opter_parms)

    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(10, 6))

    # best_x = func_opter_parms.best_x
    # func_opter_parms.parms_log['logger'].info('best x: {}'.format(best_x))

    close_log_file(logger)


    tr.used()
