# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from dramkit.datsci.stats import r2
from dramkit.datsci.utils_lgb import lgb_train
from dramkit.datsci.utils_lgb import lgb_predict
from dramkit.datsci.utils_lgb import lgb_cv_mdls
from dramkit.datsci.utils_lgb import lgb_cv_grid_search

#%%
if __name__ == '__main__':
    # 回归任务
    # data = datasets.load_boston()
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']

    valid_rate = 0.25
    X_train, X_valid, y_train, y_valid = tts(X, y, test_size=valid_rate,
                                             random_state=62)

    # 直接训练-----------------------------------------------------------------
    mdl, evals_result = lgb_train(X_train, y_train, objective='regression')
    # 验证集预测
    valid_pre, _ = lgb_predict(mdl, X_valid)
    r2_1 = r2(y_valid, valid_pre)

    # 交叉验证-----------------------------------------------------------------
    mdls, evals_results = lgb_cv_mdls(X_train, y_train, objective='regression')
    valid_pres = []
    for mdl in mdls:
        valid_pre, _ = lgb_predict(mdl, X_valid)
        valid_pres.append(valid_pre)
    valid_pre = sum(valid_pres) / len(valid_pres)
    r2_2 = r2(y_valid, valid_pre)

    # 交叉验证CV网格------------------------------------------------------------
    parms_mdl = {}
    best1, s1 = lgb_cv_grid_search(X_train, y_train, objective='regression',
                                   parms_to_opt={'num_leaves': range(3, 50, 10),
                                                 'max_depth': range(1, 10, 1),
                                        'min_data_in_leaf': range(5, 50, 5)})
    parms_mdl.update(best1)
    mdl, evals_result = lgb_train(X_train, y_train, objective='regression',
                                  parms_mdl=parms_mdl)
    valid_pre, _ = lgb_predict(mdl, X_valid)
    r2_3= r2(y_valid, valid_pre)

    best2, s2 = lgb_cv_grid_search(X_train, y_train, objective='regression',
                    parms_to_opt={'bagging_fraction': np.linspace(0.5, 1, 5),
                                            'bagging_freq': range(5, 20, 5),
                                'feature_fraction': np.linspace(0.5, 1, 5),})
    parms_mdl.update(best2)
    mdl, evals_result = lgb_train(X_train, y_train, objective='regression',
                                  parms_mdl=parms_mdl)
    valid_pre, _ = lgb_predict(mdl, X_valid)
    r2_4= r2(y_valid, valid_pre)

    best3, s3 = lgb_cv_grid_search(X_train, y_train, objective='regression',
      parms_to_opt={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
                    'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]})
    parms_mdl.update(best3)
    mdl, evals_result = lgb_train(X_train, y_train, objective='regression',
                                  parms_mdl=parms_mdl)
    valid_pre, _ = lgb_predict(mdl, X_valid)
    r2_5= r2(y_valid, valid_pre)

    print(r2_1)
    print(r2_2)
    print(r2_3)
    print(r2_4)
    print(r2_5)
