# -*- coding: utf-8 -*-

import pandas as pd
from dramkit.optimizer.ga import ga
from dramkit.optimizer.cs import cs
from dramkit.optimizer.pso import pso
from dramkit.optimizer.gwo import gwo
from dramkit.optimizer.woa import woa
from dramkit.optimizer.hho import hho
from dramkit.optimizer.alo import alo
from dramkit.optimizer.boa import boa
from dramkit.optimizer.hpsoboa import hpsoboa
from dramkit.optimizer.hcpsoboa import hcpsoboa
from dramkit.optimizer.base_funcs import TestFuncs
from dramkit.optimizer.utils_heuristic import FuncOpterInfo
from dramkit import plot_series
from dramkit import simple_logger, close_log_file
from dramkit.logtools.logger_general import get_logger


if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    # 目标函数和参数
    objf = TestFuncs.f1
    parms_func = {'func_name': objf.__name__,
                  'x_lb': -10, 'x_ub': 10, 'dim': 100, 'kwargs': {}}

    # 统一参数
    popsize = 30
    max_iter = 500

    # logger
    # logger = simple_logger()
    logger = get_logger('./log/heuristic-test.txt')
    # parms_log = {'logger': logger, 'nshow': 10}
    parms_log = {'logger': logger, 'nshow': max_iter}

    fvals = pd.DataFrame()

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

    # boa
    parms_boa = {'opter_name': 'boa',
                 'popsize': popsize, 'max_iter': max_iter,
                 'p': 0.6, 'power_exponent': 0.1, 'sensory_modality': 0.01}

    boa_parms = FuncOpterInfo(parms_func, parms_boa, parms_log)
    boa_parms = boa(objf, boa_parms)
    fvals['boa'] = boa_parms.convergence_curve

    # hpsoboa
    parms_hpsoboa = {'opter_name': 'hpsoboa',
                     'popsize': popsize, 'max_iter': max_iter,
                     'p': 0.6, 'power_exponent': 0.1, 'sensory_modality': 0.01}

    hpsoboa_parms = FuncOpterInfo(parms_func, parms_hpsoboa, parms_log)
    hpsoboa_parms = hpsoboa(objf, hpsoboa_parms)
    fvals['hpsoboa'] = hpsoboa_parms.convergence_curve

    # hcpsoboa
    parms_hcpsoboa = {'opter_name': 'hcpsoboa',
                      'popsize': popsize, 'max_iter': max_iter,
                      'p': 0.6, 'power_exponent': 0.1,
                      'sensory_modality': 0.01}

    hcpsoboa_parms = FuncOpterInfo(parms_func, parms_hcpsoboa, parms_log)
    hcpsoboa_parms = hcpsoboa(objf, hcpsoboa_parms)
    fvals['hcpsoboa'] = hcpsoboa_parms.convergence_curve

    # # alo
    # parms_alo = {'opter_name': 'alo',
    #              'popsize': popsize, 'max_iter': max_iter}

    # alo_parms = FuncOpterInfo(parms_func, parms_alo, parms_log)
    # alo_parms = alo(objf, alo_parms)
    # fvals['alo'] = alo_parms.convergence_curve


    # 参数汇总
    Results = pd.DataFrame({'ga': ga_parms.best_x,
                            'pso': pso_parms.best_x,
                            'cs': cs_parms.best_x,
                            'gwo': gwo_parms.best_x,
                            'woa': woa_parms.best_x,
                            'hho': hho_parms.best_x,
                            'boa': boa_parms.best_x,
                            'hpsoboa': hpsoboa_parms.best_x,
                            'hcpsoboa': hcpsoboa_parms.best_x,
                            # 'alo': alo_parms.best_x
                            })


    # 作图比较
    plot_series(fvals.iloc[150:, :],
                {'ga': '-', 'pso': '-', 'cs': '-', 'gwo': '-', 'woa': '-',
                 'hho': '-', 'boa': '-', 'hpsoboa': '-', 'hcpsoboa': '-',
                 # 'alo': '-'
                 },
                figsize=(10, 6))


    close_log_file(logger)


    tr.used()
