# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
from dramkit.datsci.utils_lgb import (lgb_train,
                                      lgb_predict,
                                      lgb_cv_mdls,
                                      lgb_cv_grid_search)
from dramkit.datsci.utils_ml import (vote_label_int,
                                     vote_prob_multi)

#%%
if __name__ == '__main__':
    # 多分类任务
    # data = datasets.load_iris()
    data = datasets.load_wine()
    X, y = data['data'], data['target']

    valid_rate = 0.25
    X_train, X_valid, y_train, y_valid = tts(X, y, test_size=valid_rate,
                                             random_state=62)

    # 直接训练-----------------------------------------------------------------
    mdl, evals_result = lgb_train(X_train, y_train, objective='multiclass')
    # 验证集预测
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc1 = accuracy_score(y_valid, valid_pre)

    # 交叉验证-----------------------------------------------------------------
    mdls, evals_results = lgb_cv_mdls(X_train, y_train, objective='multiclass')
    valid_pres, ps = [], []
    for mdl in mdls:
        valid_pre, p = lgb_predict(mdl, X_valid)
        valid_pres.append(valid_pre)
        ps.append(p)
    valid_pre = vote_label_int(valid_pres)
    acc21 = accuracy_score(y_valid, valid_pre)
    valid_pre = vote_prob_multi(ps)
    acc22 = accuracy_score(y_valid, valid_pre)

    # 交叉验证CV网格------------------------------------------------------------
    strt_tm = time.monotonic()
    parms_mdl = {}
    # part1
    best1, s1 = lgb_cv_grid_search(X_train, y_train, objective='multiclass',
                                   parms_to_opt={'num_leaves': range(5, 100, 5),
                                                 'max_depth': range(3, 8, 1)})
    parms_mdl.update(best1)
    mdl, evals_result = lgb_train(X_train, y_train, objective='multiclass',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc3= accuracy_score(y_valid, valid_pre)

    # part2
    best2, s2 = lgb_cv_grid_search(X_train, y_train, objective='multiclass',
                    parms_to_opt={'min_data_in_leaf': range(1, 102, 10),
                                            'max_bin': range(5, 256, 10)})
    parms_mdl.update(best2)
    mdl, evals_result = lgb_train(X_train, y_train, objective='multiclass',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc4= accuracy_score(y_valid, valid_pre)

    # part3
    best3, s3 = lgb_cv_grid_search(X_train, y_train, objective='multiclass',
                    parms_to_opt={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
                                  'bagging_fraction':  [0.6,0.7,0.8,0.9,1.0],
                                  'bagging_freq':  range(0,50,5)})
    parms_mdl.update(best3)
    mdl, evals_result = lgb_train(X_train, y_train, objective='multiclass',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc5= accuracy_score(y_valid, valid_pre)

    # part4
    best4, s4 = lgb_cv_grid_search(X_train, y_train, objective='multiclass',
      parms_to_opt={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
                    'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]})
    parms_mdl.update(best4)
    mdl, evals_result = lgb_train(X_train, y_train, objective='multiclass',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc6= accuracy_score(y_valid, valid_pre)

    # part5
    best5, s5 = lgb_cv_grid_search(X_train, y_train, objective='multiclass',
                  parms_to_opt={'min_split_gain': [0.0,0.1,0.2,0.3,0.4,0.5,
                                                   0.6,0.7,0.8,0.9,1.0],})
    parms_mdl.update(best5)
    mdl, evals_result = lgb_train(X_train, y_train, objective='multiclass',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc7= accuracy_score(y_valid, valid_pre)
    print('GridSearchCV used {}s.'.format(round(time.monotonic()-strt_tm, 6)))

    # 交叉验证-----------------------------------------------------------------
    mdls, evals_results = lgb_cv_mdls(X_train, y_train, objective='multiclass',
                                     parms_mdl_list=parms_mdl)
    valid_pres, ps = [], []
    for mdl in mdls:
        valid_pre, p = lgb_predict(mdl, X_valid)
        valid_pres.append(valid_pre)
        ps.append(p)
    valid_pre = vote_label_int(valid_pres)
    acc81 = accuracy_score(y_valid, valid_pre)
    valid_pre = vote_prob_multi(ps)
    acc82 = accuracy_score(y_valid, valid_pre)

    print(acc1)
    print(acc21)
    print(acc22)
    print(acc3)
    print(acc4)
    print(acc5)
    print(acc6)
    print(acc7)
    print(acc81)
    print(acc82)
