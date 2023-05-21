# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import metrics

from dramkit.optimizer.ga import ga
from dramkit.optimizer.cs import cs
from dramkit.optimizer.pso import pso
from dramkit.optimizer.gwo import gwo
from dramkit.optimizer.woa import woa
from dramkit.optimizer.hho import hho
from dramkit.optimizer.utils_heuristic import FuncOpterInfo
from dramkit import plot_series
from dramkit import simple_logger, close_log_file
from dramkit.logtools.logger_general import get_logger

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

#%%
def LR_cls_bin_objf(W_b, X_train=None, y_train=None, p_cut=0.5):
    '''
    逻辑回归二分类目标函数
    '''

    def sigmoid(x):
        '''sigmoid激活函数'''
        return 1.0 / (1 + np.exp(-x))

    def forward(X, W):
        '''前向传播（模型表达式）'''
        return sigmoid(np.dot(X, W))

    def XaddConst(X):
        '''X添加常数列'''
        const = np.ones((X.shape[0], 1))
        return np.concatenate((X, const), axis=1)

    W = np.array(W_b).reshape(-1, 1)

    X_train, y_train = np.array(X_train), np.array(y_train)
    Xconst = XaddConst(X_train) # X添加常数项
    # y转化为二维
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = y_train.reshape(-1, 1)

    y_pre_p = forward(Xconst, W)
    y_pre = (y_pre_p >= p_cut).astype(int)

    error = 1 - metrics.accuracy_score(y_train, y_pre)

    return error

def plot_result(data, w, b, title=None):
    plt.figure(figsize=(10, 7))
    data0 = data[data['y'] == 0]
    data1 = data[data['y'] == 1]
    plt.plot(data0['x1'], data0['x2'], 'ob', label='y=0')
    plt.plot(data1['x1'], data1['x2'], 'or', label='y=1')

    x = np.arange(data['x1'].min(), data['x1'].max(), 0.1)
    y = (-b - w[0]*x) / w[1]
    plt.plot(x, y, '-')

    plt.legend(loc=0)

    if title:
        plt.title(title)

    plt.show()

#%%
if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    #%%
    # 分类任务数据集
    data = pd.read_excel('../../datsci/_test/test_data1.xlsx')
    # data = pd.read_excel('../../datsci/_test/test_data2.xlsx')
    X = data[['x1', 'x2']]
    y = data['y']

    X_train, y_train = X, y

    #%%
    # 目标函数和参数
    objf = LR_cls_bin_objf
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -20, 'x_ub': 20, 'dim': 3,
                  'kwargs': {'X_train': X_train, 'y_train': y_train,
                             'p_cut': 0.5}}

    # 统一参数
    popsize = 10
    max_iter = 1000

    # logger
    # logger = simple_logger()
    logger = get_logger('./log/heuristic_LRcls_bin.txt')
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': max_iter}

    fvals = pd.DataFrame()

    #%%
    # ga
    parms_ga = {'opter_name': 'ga',
                'popsize': popsize, 'max_iter': max_iter,
                'p_crs': 0.7, 'p_mut': 0.1, 'n_top': 2}

    ga_parms = FuncOpterInfo(parms_func, parms_ga, parms_log)
    ga_parms = ga(objf, ga_parms)
    fvals['ga'] = ga_parms.convergence_curve

    # pso
    parms_pso = {'opter_name': 'pso',
                 'popsize': popsize, 'max_iter': max_iter,
                 'v_maxs': 5, 'w_max': 0.9, 'w_min': 0.2, 'w_fix': False,
                 'c1': 2, 'c2': 2}

    pso_parms = FuncOpterInfo(parms_func, parms_pso, parms_log)
    pso_parms = pso(objf, pso_parms)
    fvals['pso'] = pso_parms.convergence_curve

    # cs
    parms_cs = {'opter_name': 'cs',
                'popsize': popsize, 'max_iter': max_iter,
                'pa': 0.25, 'beta': 1.5, 'alpha': 0.01}

    cs_parms = FuncOpterInfo(parms_func, parms_cs, parms_log)
    cs_parms = cs(objf, cs_parms)
    fvals['cs'] = cs_parms.convergence_curve

    # gwo
    parms_gwo = {'opter_name': 'gwo',
                 'popsize': popsize, 'max_iter': max_iter}

    gwo_parms = FuncOpterInfo(parms_func, parms_gwo, parms_log)
    gwo_parms = gwo(objf, gwo_parms)
    fvals['gwo'] = gwo_parms.convergence_curve

    # woa
    parms_woa = {'opter_name': 'woa',
                 'popsize': popsize, 'max_iter': max_iter}

    woa_parms = FuncOpterInfo(parms_func, parms_woa, parms_log)
    woa_parms = woa(objf, woa_parms)
    fvals['woa'] = woa_parms.convergence_curve

    # hho
    parms_hho = {'opter_name': 'hho',
                 'popsize': popsize, 'max_iter': max_iter,
                 'beta': 1.5, 'alpha': 0.01}

    hho_parms = FuncOpterInfo(parms_func, parms_hho, parms_log)
    hho_parms = hho(objf, hho_parms)
    fvals['hho'] = hho_parms.convergence_curve

    #%%
    # 参数汇总
    Results = pd.DataFrame({'ga': ga_parms.best_x,
                            'pso': pso_parms.best_x,
                            'cs': cs_parms.best_x,
                            'gwo': gwo_parms.best_x,
                            'woa': woa_parms.best_x,
                            'hho': hho_parms.best_x})
    Results.index = ['w1', 'w2', 'b']
    Results = Results.transpose()
    print(Results)

    for n in Results.index:
        w_b = list(Results.loc[n])
        w, b = w_b[:-1], w_b[-1]
        plot_result(data, w, b, n)

    #%%
    # 作图比较
    plot_series(fvals.iloc[:, :],
                {'ga': '-', 'pso': '-', 'cs': '-', 'gwo': '-', 'woa': '-',
                 'hho': '-'},
                figsize=(10, 6))

    #%%
    close_log_file(logger)

    #%%
    tr.used()
