# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from dramkit.plottools.plot_common import plot_series
from scipy.stats.mstats import gmean
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import GradientBoostingRegressor as gbr

#%%
def get_ahr999(data, birthday, col_price='close', poly=1, fit_log=True,
               model='lr', n_fix_inv=200, fix_inv_cost='gmean',
               col_weight='volume', plot=True,
               kwargs_plot={'title': None, 'figsize': (12, 8)}):
    '''
    | AHR999指标计算
    | data中需包含['time'|'date', col_price]列
      以及VWAP用到的[col_vol, col_val]列
    | poly设置拟合模型输入X进行多项式扩展的最高次方数
    | fit_log设置拟合模型是否进行对数变换
    | model指定模型类别：'lr'为线性回归，'gbr'为gbdt回归
    | n_fix_inv设置几定投历史期数
    | fix_inv_cost设置定投成本计算方法，可选'gmean'，'vwap'
    | col_weight指定VWAP方法中的权重列（通常是成交量）
    | 参考：
    | https://www.qkl123.com/data/ahr999/btc
    | https://weibo.com/ttarticle/p/show?id=2309404441088189399138
    | https://blog.csdn.net/weixin_30521175/article/details/112435923
    '''

    if 'time' in data.columns:
        tcol = 'time'
    elif 'date' in data.columns:
        tcol = 'date'
    else:
        raise ValueError('data必须包含`time`或`date`列！')
    cols = [tcol, col_price]
    if fix_inv_cost == 'vwap':
        cols = cols + [col_weight]
    cols = list(set(cols))
    df = data.reindex(columns=cols)

    df['age'] = pd.to_datetime(df[tcol]) - pd.to_datetime(birthday)
    df['age'] = df['age'].apply(lambda x: x.days)
    if fit_log:
        df['Plog'] = np.log10(df[col_price])
        df['Alog'] = np.log10(df['age'])
        # df['Plog'] = np.log(df[col_price])
        # df['Alog'] = np.log(df['age'])
    else:
        df['Plog'] = df[col_price].copy()
        df['Alog'] = df['age'].copy()

    for p in range(1, poly+1):
        df['Alog{}'.format(p)] = df['Alog'] ** p

    Xcols = ['Alog{}'.format(p) for p in range(1, poly+1)]

    if model == 'lr':
        mdl = lr()
    elif model == 'gbr':
        mdl = gbr()

    mdl = mdl.fit(df[Xcols], df['Plog'])
    df['PlogPred'] = mdl.predict(df[Xcols])
    if fit_log:
        df['拟合值'] = np.power(10, df['PlogPred'])
        # df['拟合值'] = np.exp(df['PlogPred'])
    else:
        df['拟合值'] = df['PlogPred'].copy()

    # 定投成本
    if fix_inv_cost == 'gmean': # 几何平均值
        df['定投{}成本'.format(n_fix_inv)] = df[col_price].rolling(n_fix_inv).apply(gmean)
    elif fix_inv_cost == 'vwap':
        df['tmp_val'] = df[col_price] * df[col_weight]
        df['定投{}成本'.format(n_fix_inv)] = df['tmp_val'].rolling(n_fix_inv).sum() / \
                                  df[col_weight].rolling(n_fix_inv).sum()
        df.drop('tmp_val', axis=1, inplace=True)

    df['Price/Cost'] = df[col_price] / df['定投{}成本'.format(n_fix_inv)]
    df['Price/Pred'] = df[col_price] / df['拟合值']
    df['AHR999'] = df['Price/Cost'] * df['Price/Pred']
    if fit_log:
        df['AHR999'] = np.log10(df['AHR999'])
        # df['AHR999'] = np.log(df['AHR999'])

    if plot:
        kwargs_plot_ = kwargs_plot.copy()
        title = kwargs_plot_['title']
        del kwargs_plot_['title']
        title = 'AHR999' if title is None else title + '_AHR999'
        yscales = ['log' if fit_log else 'linear', 'linear', 'linear']
        plot_series(df, {col_price: '-k', '拟合值': '--r',
                         '定投{}成本'.format(n_fix_inv): '--g'},
                    cols_styl_up_right={'AHR999': '-b'},
                    cols_styl_low_left={'Price/Cost': '-g',
                                        'Price/Pred': '-b'},
                    xparls_info={'Price/Cost': [(1, 'm', '-', 1.5)]},
                    yscales=yscales, title=title,
                    **kwargs_plot_
                    )

    return df['AHR999']

#%%
if __name__ == '__main__':
    import time
    from dramkit.fintools.load_his_data import load_btc_daily
    from dramkit.fintools.load_his_data import load_index_daily
    from dramkit.fintools.load_his_data import load_index_futures_daily

    strt_tm = time.time()

    #%%
    data = load_btc_daily()[['time', 'close', 'vol']]
    data = data[data['time'] >= '2011-03-03']

    data['AHR999'] = get_ahr999(data, '2009-01-03', fit_log=True)
    data['AHR999vwap'] = get_ahr999(data, '2009-01-03', fit_log=True,
                                    fix_inv_cost='vwap', col_weight='vol')

    plot_series(data, {'close': '-k'},
                cols_styl_up_right={'AHR999': '-b', 'AHR999vwap': '-y'},
                yscales=['log', 'linear'])

    #%%
    code = 'IF'
    birthday = '2010-01-01'
    data = load_index_futures_daily(code+'9999')[['date', 'close']]
    data.set_index('date', drop=False, inplace=True)
    data['AHR999'] = get_ahr999(data, birthday, n_fix_inv=150, fit_log=False,
                                poly=1, model='lr', plot=True,
                                kwargs_plot={'title': code})

    plot_series(data, {'close': '-k'},
                cols_styl_up_right={'AHR999': '-b'},
                yscales=['linear', 'linear'],
                xparls_info={'AHR999': [(0.85, 'b', '-', 1.5),
                                        (1.0, 'b', '-', 1.5),
                                        (1.2, 'b', '-', 1.5)]},
                title=code)

    #%%
    code = '000300'
    birthday = '2010-01-01'
    data = load_index_daily(code)[['time', 'close', 'volume', 'money']]
    data.set_index('time', drop=False, inplace=True)
    data['AHR999'] = get_ahr999(data, birthday, n_fix_inv=150, fit_log=False,
                                poly=2, model='gbr', plot=True,
                                kwargs_plot={'title': code})
    data['AHR999vwap'] = get_ahr999(data, birthday, n_fix_inv=150, fit_log=False,
                                poly=2, model='gbr', fix_inv_cost='vwap',
                                col_weight='money', plot=True,
                                kwargs_plot={'title': code})

    plot_series(data, {'close': '-k'},
                cols_styl_up_right={'AHR999': '-b', 'AHR999vwap': '-y'},
                yscales=['linear', 'linear'],
                xparls_info={'AHR999': [(0.85, 'b', '-', 1.5),
                                        (1.0, 'b', '-', 1.5),
                                        (1.2, 'b', '-', 1.5)]},
                title=code)

    #%%
    code = 'IC'
    birthday = '2010-01-01'
    data = load_index_futures_daily(code+'9999')[['date', 'close']]
    data.set_index('date', drop=False, inplace=True)
    data['AHR999'] = get_ahr999(data, birthday, n_fix_inv=100, fit_log=False,
                                poly=2, model='lr', plot=True,
                                kwargs_plot={'title': code})

    plot_series(data, {'close': '-k'},
                cols_styl_up_right={'AHR999': '-b'},
                yscales=['linear', 'linear'],
                xparls_info={'AHR999': [(0.85, 'b', '-', 1.5),
                                        (1.0, 'b', '-', 1.5),
                                        (1.2, 'b', '-', 1.5)]},
                title=code)

    #%%
    code = '000905'
    birthday = '2010-01-01'
    data = load_index_daily(code)[['time', 'close']]
    data.set_index('time', drop=False, inplace=True)
    data['AHR999'] = get_ahr999(data, birthday, n_fix_inv=150, fit_log=False,
                                poly=1, model='lr', plot=True,
                                kwargs_plot={'title': code})

    plot_series(data, {'close': '-k'},
                cols_styl_up_right={'AHR999': '-b'},
                yscales=['linear', 'linear'],
                xparls_info={'AHR999': [(0.85, 'b', '-', 1.5),
                                        (1.0, 'b', '-', 1.5),
                                        (1.2, 'b', '-', 1.5)]},
                title=code)

    #%%
    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
