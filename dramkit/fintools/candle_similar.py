# -*- coding: utf-8 -*-

from dtw import dtw # https://dynamictimewarping.github.io/


def candle_similar(df1, df2, cols, std01=True):
    '''
    计算K线相似度，df1和df2应包含K线数据指定列cols
    '''
    if std01:
        df1, df2 = df1.copy(), df2.copy()
        for col in cols:
            vmin1, vmax1 = df1[col].min(), df1[col].max()
            df1[col] = (df1[col] - vmin1) / (vmax1 - vmin1)
            vmin2, vmax2 = df2[col].min(), df2[col].max()
            df2[col] = (df2[col] - vmin2) / (vmax2 - vmin2)
    alignment = dtw(df1[cols], df2[cols])
    sim = alignment.distance
    sim_std = alignment.normalizedDistance
    return sim_std, sim
    
    
if __name__ == '__main__':
    import time
    import pandas as pd
    from dramkit import load_csv, plot_series
    from dramkit.plottools.plot_candle import plot_candle
    
    strt_tm = time.time()
    
    # test
    df1 = pd.DataFrame({'close':
                        [1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1]})
    df2 = pd.DataFrame({'close':
                        [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2]})
    df3 = pd.DataFrame({'close': [0.8, 1.5, 0, 1.2, 0, 0, 0.6, 1, 1.2, 0, 0,
                                  1, 0.2, 2.4, 0.5, 0.4]})
    df4 = pd.DataFrame({'close': [1, 2, 3, 4]})
    df5 = pd.DataFrame({'close': [5, 4, 3, 2]})
    plot_series(df1, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)
    plot_series(df2, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)
    plot_series(df3, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)
    plot_series(df4, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)
    plot_series(df5, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)
    print(candle_similar(df1, df2, ['close']))
    print(candle_similar(df1, df3, ['close']))
    print(candle_similar(df4, df5, ['close']))
    
    
    # 50ETF日线行情
    fpath = '../test/510050_daily_pre_fq.csv'
    df = load_csv(fpath)
    df['time'] = df['date']
    df.set_index('date', drop=True, inplace=True)
    df = df.reindex(columns=['time', 'open', 'low', 'high', 'close', 'volume'])
    
    df1 = df.iloc[-50:-42, :]
    df2 = df.iloc[-40:-32:, :]
    plot_candle(df1, args_ma=None, args_boll=None,
                plot_below=None, figsize=(6, 4))
    plot_candle(df2, args_ma=None, args_boll=None,
                plot_below=None, figsize=(6, 4))
    plot_series(df1, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)
    plot_series(df2, {'close': '.-k'}, figsize=(6, 3), n_xticks=4)    
    
    print(candle_similar(df1, df2, ['close', 'high', 'low', 'open']))   


    
    print(f'used time: {round(time.time()-strt_tm, 6)}s.')
    
    
    
    
    
    
    
