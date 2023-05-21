# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from dramkit.datsci.find_maxmin import (find_maxmin,
                                        check_maxmins)
from dramkit.datsci.zigzag import zigzag
from dramkit import TimeRecoder, load_csv
from dramkit.plottools.plot_common import plot_series 

#%%
if __name__ == '__main__':

    tr = TimeRecoder()
    
    #%%
    def plot_res(df, col):
        plot_series(df, {col: ('.-k', None)},
                    cols_to_label_info={
                        col: [
                            ['label', (-1, 1), ('r^', 'gv'), None],
                            ]},
                    figsize=(9, 4))
        plot_series(df, {col: ('.-k', None)},
                    cols_to_label_info={
                        col: [
                            ['zigzag', (-1, 1), ('r^', 'gv'), None],
                            ]},
                    figsize=(9, 4))
    
    #%%
    # '''
    arr = [1, 1, 1.3, 1.2, 2, 3, 4.8, 4.7, 5, 5]
    df1 = pd.DataFrame({'col': arr})
    df1['label'] = find_maxmin(df1['col'])
    df1['zigzag'] = zigzag(df1, 'col', 'col',
                           t_min_up=2, t_min_down=2,
                           up_min_pct=-1, down_min_pct=1
                           )
    df1['zigzag'] = df1['zigzag'].replace({0.5: 1, -0.5: -1})
    nosame = (df1['label'] != df1['zigzag']).sum()
    print(nosame)
    plot_res(df1, 'col')

    df2 = pd.DataFrame({'col': arr[::-1]})
    df2['label'] = find_maxmin(df2['col'])
    df2['zigzag'] = zigzag(df2, 'col', 'col',
                           t_min_up=2, t_min_down=2,
                           up_min_pct=-1, down_min_pct=1
                           )
    df2['zigzag'] = df2['zigzag'].replace({0.5: 1, -0.5: -1})
    nosame = (df2['label'] != df2['zigzag']).sum()
    print(nosame)
    plot_res(df2, 'col')
    # '''

    #%%
    # '''
    # 二次曲线叠加正弦余弦-------------------------------------------------------
    N = 200
    t = np.linspace(0, 1, N)
    s = 6*t*t + np.cos(10*2*np.pi*t*t) + np.sin(6*2*np.pi*t)
    df = pd.DataFrame(s, columns=['test'])

    # t_min = None
    t_min = 5
    df['label'] = find_maxmin(df['test'], t_min=t_min, min_dif_val=4, t_max=10)
    df['zigzag'] = zigzag(df, 'test', 'test',
                          t_min_up=t_min, t_min_down=t_min,
                          t_max_up=10, t_max_down=10,
                          up_min_val=4, down_min_val=-4,
                          up_min_pct=None, down_min_pct=None
                          )
    df['zigzag'] = df['zigzag'].replace({0.5: 1, -0.5: -1})
    nosame = (df['label'] != df['zigzag']).sum()
    print(nosame)
    plot_res(df, 'test')
    # '''
        
    #%%
    # '''
    # 趋势线叠加正弦余弦---------------------------------------------------------
    N = 200
    t = np.linspace(0, 1, N)
    trend = 6 * t
    circle1 = np.cos(10*2*np.pi*t*t)
    circle2 = np.sin(6*2*np.pi*t)
    # circle2 = 0 * t
    # np.random.seed(5262)
    noise = np.random.randn(len(t)) / 5
    s = trend + circle1 + circle2 + noise
    # df = pd.DataFrame({'trend': trend,
    #                    'circle1': circle1,
    #                    'circle2': circle2,
    #                    'noise': noise,
    #                    'series': s})
    df = load_csv('./find_maxmin_zigzag_test1.csv')

    t_min = 3
    df['label'] = find_maxmin(df['series'],
                              t_min=t_min,
                              min_dif_val=1,
                              t_max=10)
    df['zigzag'] = zigzag(df, 'series', 'series',
                          t_min_up=t_min, t_min_down=t_min,
                          t_max_up=10, t_max_down=10,
                          up_min_val=1, down_min_val=-1,
                          up_min_pct=None, down_min_pct=None
                          )
    df['zigzag'] = df['zigzag'].replace({0.5: 1, -0.5: -1})
    nosame = (df['label'] != df['zigzag']).sum()
    print(nosame)
    plot_res(df, 'series')
    # '''

    #%%
    # '''
    # 50ETF日线行情------------------------------------------------------------
    fpath = '../../_test/510500.SH_daily_qfq.csv'
    his_data = load_csv(fpath)
    his_data.set_index('date', drop=False, inplace=True)

    # N = his_data.shape[0]
    N = 200
    col = 'close'
    df = his_data.iloc[-N:, :][[col]].copy()
    
    t_min = 3
    min_dif_pct = None
    df['label'] = find_maxmin(df[col], t_min=t_min,
                              min_dif_pct=min_dif_pct
                              )
    from dramkit import isnull
    if isnull(min_dif_pct):
        up_min_pct, down_min_pct = -1.0, 1.0
    else:
        up_min_pct, down_min_pct = min_dif_pct, -min_dif_pct
    df['zigzag'] = zigzag(df, col, col,
                          t_min=t_min, 
                          min_pct=up_min_pct
                          )
    df['zigzag'] = df['zigzag'].replace({0.5: 1, -0.5: -1})
    nosame = (df['label'] != df['zigzag']).sum()
    print(nosame)
    plot_res(df, col)
    
    for c in ['label', 'zigzag']:
        OK, e = check_maxmins(df, col, c)
        if OK:
            print('%s极值点排列正确！'%c)
        else:
            print('%s极值点排列错误:'%c, e)
    # '''

    #%%
    tr.used()
