# -*- coding: utf-8 -*-

import numpy as np
from dramkit.logtools.logger_general import get_logger


class FuncOpterInfo(object):
    '''保存函数参数及优化过程信息'''

    def __init__(self, parms_func={}, parms_opter={}, parms_log={}):
        '''
        初始化目标函数和优化器信息
        
        Parameters
        ----------
        parms_func : dict
            目标函数信息，默认应包含'func_name', 'x_lb', 'x_ub', 'dim'
        parms_opter : dcit
            优化函数需要用到的参数信息，默认应包含'opter_name', 'popsize', 'max_iter'
        parms_log : 
            寻优过程中控制打印或日志记录的参数，默认应包含'logger', 'nshow'
        '''

        # 目标函数信息
        parms_func_default = {'func_name': '', 'x_lb': None, 'x_ub': None,
                              'dim': None, 'kwargs': {}}
        parms_loss = {x: parms_func_default[x] \
                                for x in parms_func_default.keys() if \
                                    x not in parms_func.keys()}
        parms_func.update(parms_loss)
        self.parms_func = parms_func

        # 优化算法参数
        parms_opter_default = {'opter_name': '', 'popsize': 20, 'max_iter': 100}
        parms_loss = {x: parms_opter_default[x] \
                                for x in parms_opter_default.keys() if \
                                    x not in parms_opter.keys()}
        parms_opter.update(parms_loss)
        self.parms_opter = parms_opter

        # 日志参数
        parms_log_default = {'logger': get_logger(), 'nshow': 10}
        parms_loss = {x: parms_log_default[x] \
                                for x in parms_log_default.keys() if \
                                    x not in parms_log.keys()}
        parms_log.update(parms_loss)
        self.parms_log = parms_log

        # 优化过程和结果
        self.__best_val = None # 全局最优值
        self.__best_x = [] # 全局最优解
        self.__convergence_curve = [] # 收敛曲线（每轮最优）
        self.__convergence_curve_mean = [] # 收敛曲线（每轮平均）
        self.__start_time = None # 开始时间
        self.__end_time = None # 结束时间
        self.__exe_time = None # 优化用时（单位秒）

    @property
    def best_val(self):
        '''全局最目标优值'''
        return self.__best_val

    def set_best_val(self, val):
        '''全局最目标优值设置'''
        self.__best_val = val

    @property
    def best_x(self):
        '''全局最优解'''
        return self.__best_x

    def set_best_x(self, x):
        '''全局最优解设置'''
        self.__best_x = x

    @property
    def convergence_curve(self):
        '''收敛曲线（每轮最优）'''
        return self.__convergence_curve

    def set_convergence_curve(self, curve):
        '''收敛曲线（每轮最优）设置'''
        self.__convergence_curve = curve

    @property
    def convergence_curve_mean(self):
        '''收敛曲线（每轮平均）'''
        return self.__convergence_curve_mean

    def set_convergence_curve_mean(self, curve):
        '''收敛曲线（每轮平均）设置'''
        self.__convergence_curve_mean = curve

    @property
    def start_time(self):
        '''开始时间'''
        return self.__start_time

    def set_start_time(self, t):
        '''开始时间设置'''
        self.__start_time = t

    @property
    def end_time(self):
        '''结束时间'''
        return self.__end_time

    def set_end_time(self, t):
        '''结束时间设置'''
        self.__end_time = t

    @property
    def exe_time(self):
        '''优化用时（单位秒）'''
        return self.__exe_time

    def set_exe_time(self, t):
        '''优化用时（单位秒）设置'''
        self.__exe_time = t


def rand_init(popsize, dim, lb, ub):
    '''
    自变量随机初始化

    Parameters
    ----------
    popsize : int
        需要初始化的种群数（样本数）
    dim : int
        自变量维度数，dim的值应与lb和ub的长度相等
    lb : list
        自变量每个维度取值下界
    ub : list
        自变量每个维度取值上界

    Returns
    -------
    pos : np.ndarray
        随机初始化结果，形状为popsize * dim
    '''

    pos = np.zeros((popsize, dim))
    for i in range(dim):
        pos[:, i] = np.random.uniform(lb[i], ub[i], popsize)

    return pos


def sort_population(population, fvals):
    '''
    个体排序：按fvals值从小到大对population进行排序

    Parameters
    ----------
    population : np.ndarray
        所有个体位置（所有解）
    fvals : np.array
        所有个体值列表

    Returns
    -------
    population : np.ndarray
        排序后的种群
    fvals : np.array
        排序后的个体值列表
    '''

    sorted_indices = np.argsort(fvals)
    population = population[sorted_indices]
    fvals = fvals[sorted_indices]

    return population, fvals
