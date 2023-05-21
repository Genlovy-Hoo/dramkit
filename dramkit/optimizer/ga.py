# -*- coding: utf-8 -*-

import time
import random
import numpy as np
from dramkit.gentools import isnull
from dramkit.optimizer.utils_heuristic import rand_init


def _calculate_cost(objf, population, popsize, x_lb, x_ub, **kwargs):
    '''
    计算种群中每个个体的函数值

    Parameters
    ----------
    objf : function
        目标函数，接受每个个体以及kwargs为参数
    population : np.ndarray
        所有个体位置（所有解）
    popsize : int
        种群个体数量
    x_lb, x_ub : list
        取值上下边界

    Returns
    -------
    fvals : np.array
        个体函数值列表
    '''

    fvals = np.full(popsize, np.inf)
    for i in range(0,popsize):
        # 越界处理
        population[i] = np.clip(population[i], x_lb, x_ub)
        # 个体值计算
        fvals[i] = objf(population[i, :], **kwargs)

    return fvals


def _sort_population(population, fvals):
    '''
    个体排序：较优值排在前面

    Parameters
    ----------
    population : np.ndarray
        所有个体位置（所有解）
    fvals : np.arran
        所有个体值列表

    Returns
    -------
    population : np.array
        排序后的种群
    fvals : np.array
        排序后的个体值列表
    '''

    sortedIndices = fvals.argsort()
    population = population[sortedIndices]
    fvals = fvals[sortedIndices]

    return population, fvals


def _crossover_populaton(population, fvals, popsize, p_crs, n_top, dim):
    '''
    群体交叉

    Parameters
    ----------
    population : np.ndarray
        所有个体位置（所有解）
    fvals : np.array
        所有个体值列表
    popsize : int
        种群个体数量
    p_crs : float
        交叉概率
    n_top : int
        最优个体保留数（不进行交叉变异的最优个体数）

    Returns
    -------
    new_population : np.ndarray
        新种群位置（新解）
    '''

    # 新种群初始化
    new_population = population.copy()

    for i in range(n_top, popsize-1, 2):
        # 轮盘赌法选择待交叉个体
        parent1, parent2 = _pair_selection(population, fvals, popsize)
        parentsCrossoverProbability = random.uniform(0.0, 1.0)
        if parentsCrossoverProbability < p_crs:
            offspring1, offspring2 = _crossover(dim, parent1, parent2)
            # 更新交叉后的个体
            new_population[i] = offspring1
            new_population[i+1] = offspring2

    return new_population


def _mutate_populaton(population, popsize, p_mut, n_top, x_lb, x_ub):
    '''
    群体变异

    Parameters
    ----------
    population : np.ndarray
        所有个体位置（所有解）
    popsize : int
        种群个体数量
    p_mut : float
        个体变异概率
    n_top : int
        最优个体保留数（不进行交叉变异的最优个体数）
    x_lb, x_ub : list
        取值上下边界

    Returns
    -------
    new_population : np.ndarray
        新种群位置（新解）
    '''

    new_population = population.copy()
    for i in range(n_top, popsize):
        # 变异操作
        offspringMutationProbability = random.uniform(0.0, 1.0)
        if offspringMutationProbability < p_mut:
            offspring = _mutation(population[i], len(population[i]), x_lb, x_ub)
            new_population[i] = offspring
    return new_population


def _pair_selection(population, fvals, popsize):
    '''
    轮盘赌法选择交叉个体对

    Parameters
    ----------
    population : np.ndarray
        所有个体位置（所有解）
    fvals : np.array
        所有个体值列表
    popsize : int
        种群个体数量

    Returns
    -------
    parent1, parent2 : np.array
        被选中的两个个体    
    '''

    parent1Id = _roulette_wheel_selection_id(fvals, popsize)
    parent1 = population[parent1Id].copy()

    parent2Id = _roulette_wheel_selection_id(fvals, popsize)
    parent2 = population[parent2Id].copy()

    return parent1, parent2


def _roulette_wheel_selection_id(fvals, popsize):
    '''
    轮盘赌法：个体函数值越小（最小值问题），越容易被选中

    Parameters
    ----------
    fvals : np.array
        所有个体值列表
    popsize : int
        种群个体数量

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
    for individual_id in range(popsize):
        current += reverseScores[individual_id]
        if current > pick:
            return individual_id

def _crossover(individualLength, parent1, parent2):
    '''
    两个个体交叉操作

    Parameters
    ----------
    individualLength : int
        个体长度（维度）
    parent1, parent2 : np.array
        待交叉个体

    Returns
    -------
    offspring1, offspring2 : np.array
        交叉操作后的两个新个体
    '''

    # 选择交叉位置
    crossover_point = random.randint(0, individualLength-1)
    # 以交叉位置为切分，新个体的前半部分取个体1，后半部分取个体2
    offspring1 = np.concatenate([parent1[0:crossover_point],
                                 parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[0:crossover_point],
                                 parent1[crossover_point:]])

    return offspring1, offspring2


def _mutation(offspring, individualLength, x_lb, x_ub):
    '''
    个体变异操作

    Parameters
    ----------
    offspring : np.array
        待变异个体
    individualLength : int
        个体长度
    x_lb, x_ub : list
        取值上下边界

    Returns
    -------
    offspring : np.array
        返回变异后的个体
    '''

    # 随机选择变异位置，随机取变异值
    mutationIndex = random.randint(0, individualLength-1)
    mutationValue = random.uniform(x_lb[mutationIndex], x_ub[mutationIndex])
    offspring[mutationIndex] = mutationValue
    return offspring


def _clear_dups(population, dim, x_lb, x_ub):
    '''
    替换重复个体

    Parameters
    ----------
    population : np.ndarray
        所有个体位置（所有解）
    x_lb, x_ub : list
        取值上下边界

    Returns
    -------
    new_population : np.ndarray
        随机替换重复值后的新种群
    '''

    new_population = np.unique(population, axis=0)
    oldLen = len(population)
    newLen = len(new_population)
    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newIndividuals = rand_init(nDuplicates, dim, x_lb, x_ub)
        new_population = np.append(new_population, newIndividuals, axis=0)

    return new_population


def ga(objf, func_opter_parms):
    '''
    遗传算法(Genetic Algorithm) GA（实数编码）
    
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
        |     p_crs: 交叉概率
        |     p_mut: 变异概率
        |     n_top: 每一轮（代）保留的最优个体数
        | parms_log: 日志参数信息dict，key须包含:
        |     logger: 日志记录器
        |     nshow: 若为整数，则每隔nshow轮日志输出当前最优目标函数值

    Returns
    -------
    func_opter_parms : FuncOpterInfo
        更新优化过程之后的func_opter_parms

    References
    ----------
    - https://www.jianshu.com/p/8c0260c21af4
    - https://github.com/7ossam81/EvoloPy
    '''

    # 参数提取
    opter_name = func_opter_parms.parms_opter['opter_name']
    if opter_name == '' or isnull(opter_name):
        opter_name  = 'ga'
    func_opter_parms.parms_opter['opter_name'] = opter_name
    # 目标函数参数
    x_lb = func_opter_parms.parms_func['x_lb']
    x_ub = func_opter_parms.parms_func['x_ub']
    dim = func_opter_parms.parms_func['dim']
    kwargs = func_opter_parms.parms_func['kwargs']
    # 优化器参数
    popsize = func_opter_parms.parms_opter['popsize']
    max_iter = func_opter_parms.parms_opter['max_iter']
    p_crs = func_opter_parms.parms_opter['p_crs']
    p_mut = func_opter_parms.parms_opter['p_mut']
    n_top = func_opter_parms.parms_opter['n_top']
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

    # 全局最优解和全局最优值
    gBest = np.zeros(dim)
    gBestVal = float('inf')

    population = rand_init(popsize, dim, x_lb, x_ub) # 样本（个体）随机初始化
    fvals = np.random.uniform(0.0, 1.0, popsize) # 个体函数值

    convergence_curve = np.zeros(max_iter) # 全局最优值
    convergence_curve_mean = np.zeros(max_iter) # 平均值


    for l in range(max_iter):

        # 计算个体值
        fvals = _calculate_cost(objf, population, popsize, x_lb, x_ub, **kwargs)

        # 个体排序
        population, fvals = _sort_population(population, fvals)

        # 最优解纪录
        gBestVal = fvals[0]
        gBest = population[0]

        # 交叉
        population = _crossover_populaton(population, fvals, popsize, p_crs, n_top,
                                          dim)
        # 变异
        population = _mutate_populaton(population, popsize, p_mut, n_top,
                                       x_lb, x_ub)
        # 重复值处理
        population = _clear_dups(population, dim, x_lb, x_ub)

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
    parms_opter = {'opter_name': 'ga-test',
                   'popsize': 20, 'max_iter': 1000,
                   'p_crs': 0.7, 'p_mut': 0.1, 'n_top': 2}
    # logger = simple_logger()
    logger = get_logger('./_test/log/ga_test.txt')
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': 100}

    func_opter_parms = FuncOpterInfo(parms_func, parms_opter, parms_log)
    func_opter_parms = ga(objf, func_opter_parms)

    vals = pd.DataFrame({'fval_best': func_opter_parms.convergence_curve,
                         'fval_mean': func_opter_parms.convergence_curve_mean})
    plot_series(vals, {'fval_best': '-r', 'fval_mean': '-b'}, figsize=(10, 6),
                title='GA优化目标函数值收敛过程')

    best_x = func_opter_parms.best_x
    func_opter_parms.parms_log['logger'].info('best x: {}'.format(best_x))

    close_log_file(logger)


    tr.used()
