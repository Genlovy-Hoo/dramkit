# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, roc_auc_score
from dramkit.datsci.utils_lgb import lgb_cv_mdls
from dramkit.datsci.utils_lgb import lgb_train
from dramkit.datsci.utils_lgb import lgb_predict
from dramkit.datsci.utils_lgb import lgb_cv_grid_search
from dramkit.datsci.utils_ml import vote_label_int
from dramkit.datsci.utils_ml import vote_prob_bin_pcut

#%%
if __name__ == '__main__':
    # 二分类任务
    data = datasets.load_breast_cancer()
    X, y = data['data'], data['target']

    valid_rate = 0.25
    X_train, X_valid, y_train, y_valid = tts(X, y, test_size=valid_rate,
                                             random_state=62)

    # 直接训练-----------------------------------------------------------------
    mdl, evals_result = lgb_train(X_train, y_train, objective='binary')
    # 验证集预测
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc1 = accuracy_score(y_valid, valid_pre)

    # 交叉验证-----------------------------------------------------------------
    mdls, evals_results = lgb_cv_mdls(X_train, y_train, objective='binary')
    valid_pres, ps = [], []
    for mdl in mdls:
        valid_pre, p = lgb_predict(mdl, X_valid)
        valid_pres.append(valid_pre)
        ps.append(p)
    valid_pre = vote_label_int(valid_pres)
    acc21 = accuracy_score(y_valid, valid_pre)
    valid_pre = vote_prob_bin_pcut(ps)
    acc22 = accuracy_score(y_valid, valid_pre)

    # 交叉验证CV网格------------------------------------------------------------
    parms_mdl = {'boosting_type': 'gbdt',
                 'objective': 'binary',
                 'metric': 'auc',
                 'nthread': 4,
                 'learning_rate': 0.1}
    # part1
    best1, s1 = lgb_cv_grid_search(X_train, y_train, objective='binary',
                                   parms_to_opt={'num_leaves': range(5, 100, 5),
                                                 'max_depth': range(3, 8, 1)})
    parms_mdl.update(best1)
    mdl, evals_result = lgb_train(X_train, y_train, objective='binary',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc3= accuracy_score(y_valid, valid_pre)

    # part2
    best2, s2 = lgb_cv_grid_search(X_train, y_train, objective='binary',
                    parms_to_opt={'min_data_in_leaf': range(1, 102, 10),
                                            'max_bin': range(5, 256, 10)})
    parms_mdl.update(best2)
    mdl, evals_result = lgb_train(X_train, y_train, objective='binary',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc4= accuracy_score(y_valid, valid_pre)

    # part3
    best3, s3 = lgb_cv_grid_search(X_train, y_train, objective='binary',
                    parms_to_opt={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
                                  'bagging_fraction':  [0.6,0.7,0.8,0.9,1.0],
                                  'bagging_freq':  range(0,50,5)})
    parms_mdl.update(best3)
    mdl, evals_result = lgb_train(X_train, y_train, objective='binary',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc5= accuracy_score(y_valid, valid_pre)

    # part4
    best4, s4 = lgb_cv_grid_search(X_train, y_train, objective='binary',
      parms_to_opt={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
                    'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]})
    parms_mdl.update(best4)
    mdl, evals_result = lgb_train(X_train, y_train, objective='binary',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc6= accuracy_score(y_valid, valid_pre)

    # part5
    best5, s5 = lgb_cv_grid_search(X_train, y_train, objective='binary',
                  parms_to_opt={'min_split_gain': [0.0,0.1,0.2,0.3,0.4,0.5,
                                                   0.6,0.7,0.8,0.9,1.0],})
    parms_mdl.update(best5)
    mdl, evals_result = lgb_train(X_train, y_train, objective='binary',
                                  parms_mdl=parms_mdl)
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc7= accuracy_score(y_valid, valid_pre)


    mdl, evals_result = lgb_train(X_train, y_train, objective='binary',
                                  parms_mdl={'bagging_fraction': 0.7,
                                             'bagging_freq': 30,
                                             'feature_fraction': 0.8,
                                             'lambda_l1': 0.1,
                                             'lambda_l2': 0.0,
                                             'max_bin': 255,
                                             'max_depth': 4,
                                             'min_data_in_leaf': 81,
                                             'min_split_gain': 0.1,
                                             'num_leaves': 10,
                                             'learning_rate': 0.01})
    valid_pre, p = lgb_predict(mdl, X_valid)
    acc8 = accuracy_score(y_valid, valid_pre)
    auc = roc_auc_score(y_valid, p)

    print(acc1)
    print(acc21)
    print(acc22)
    print(acc3)
    print(acc4)
    print(acc5)
    print(acc6)
    print(acc7)
    print(acc8)
    print(auc)
