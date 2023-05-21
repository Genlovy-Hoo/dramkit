# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
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
from dramkit.datsci.preprocess import scale_skl
from dramkit.datsci.stats import mape

#%%
def gbc_objf(superParams, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None):
    '''
    构造gbdt分类模型目标函数（适应度函数）
    '''
    max_depth = int(superParams[0])
    subsample = superParams[1]
    min_samples_leaf = int(superParams[2])
    min_samples_split = int(superParams[3])
    mdl = GBC(max_depth=max_depth,
              subsample=subsample,
              min_samples_leaf=min_samples_leaf,
              min_samples_split=min_samples_split)
    mdl = mdl.fit(Xtrain, Ytrain)
    Ypre = mdl.predict(Xtest)
    error = 1 - metrics.accuracy_score(Ytest, Ypre)
    return error


def gbr_objf(superParams, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None,
             **kwargs):
    '''
    构造gbdt回归模型目标函数（适应度函数）
    '''
    max_depth = int(superParams[0])
    subsample = superParams[1]
    min_samples_leaf = int(superParams[2])
    min_samples_split = int(superParams[3])
    mdl = GBC(max_depth=max_depth,
              subsample=subsample,
              min_samples_leaf=min_samples_leaf,
              min_samples_split=min_samples_split, **kwargs)
    mdl = mdl.fit(Xtrain, Ytrain)
    Ypre = mdl.predict(Xtest)
    vMAPE = mape(Ytest, Ypre)
    return vMAPE

#%%
if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    #%%
    # 分类任务数据集
    data_cls = datasets.load_iris()
    X_cls = pd.DataFrame(data_cls['data'], columns=data_cls.feature_names)
    Y_cls = pd.Series(data_cls['target'])

    Xtrain_cls, Xtest_cls, Ytrain_cls, Ytest_cls = tts(X_cls, Y_cls,
                                        test_size=0.4, random_state=5262)
    Xtrain_cls, [Xtest_cls], _ = scale_skl(Xtrain_cls, [Xtest_cls])


    # 回归任务数据集
    # data_reg = datasets.load_boston()
    data_reg = datasets.load_diabetes()
    X_reg = pd.DataFrame(data_reg['data'], columns=data_reg.feature_names)
    Y_reg = pd.Series(data_reg['target'])

    Xtrain_reg, Xtest_reg, Ytrain_reg, Ytest_reg = tts(X_reg, Y_reg,
                                        test_size=0.4, random_state=5262)
    Xtrain_reg, [Xtest_reg], _ = scale_skl(Xtrain_reg, [Xtest_reg])

    #%%
    # 目标函数和参数
    objf = gbr_objf
    parms_func = {'func_name': objf.__name__,
                  'x_lb': [1, 0.01, 1, 2], 'x_ub': [10, 1.0, 10, 10], 'dim': 4,
                  'kwargs': {'Xtrain': Xtrain_reg, 'Ytrain': Ytrain_reg,
                             'Xtest': Xtest_reg, 'Ytest': Ytest_reg,
                             'n_estimators': 100,}}
    # objf = gbc_objf
    # parms_func = {'func_name': objf.__name__,
    #               'x_lb': 0.01, 'x_ub': 100, 'dim': 2,
    #               'kwargs': {'Xtrain': Xtrain_cls, 'Ytrain': Ytrain_cls,
    #                           'Xtest': Xtest_cls, 'Ytest': Ytest_cls}}

    # 统一参数
    popsize = 5
    max_iter = 10

    # logger
    # logger = simple_logger()
    logger = get_logger('./log/heuristic_gbdt.txt')
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
    # 作图比较
    plot_series(fvals.iloc[:, :],
                {'ga': '-', 'pso': '-', 'cs': '-', 'gwo': '-', 'woa': '-',
                  'hho': '-'},
                figsize=(10, 6))

    #%%
    close_log_file(logger)

    #%%
    tr.used()
