# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from dramkit.gentools import (isnull,
                              cal_pct,
                              get_first_appear)
from dramkit.logtools.utils_logger import logger_show


def zigzag(df: pd.DataFrame,
           high_col: str = 'high',
           low_col: str = 'low',
           t_min: int = None,
           t_max: int = np.inf,
           min_pct: float = None,
           min_val: float = None,
           max_pct: float = None,
           max_val: float = None,
           t_min_up: int = None,
           t_min_down: int = None,
           t_max_up: int = None,
           t_max_down: int = None,
           up_min_pct: float = None, # 1/100,
           up_min_val: float = None,
           up_max_pct: float = None,
           up_max_val: float = None,
           down_min_pct: float = None, # -1/100,
           down_min_val: float = None,
           down_max_pct: float = None,
           down_max_val: float = None,
           pct_v00: float = 1.0,
           logger: logging.Logger = None
           ) -> pd.Series:
    '''
    ZigZag转折点

    Parameters
    ----------
    df : pd.DataFrame
        需包含[high_col, low_col]列
    high_col : str
        确定zigzag高点的数据列名
    low_col : str
        确定zigzag低点的数据列名
    t_min : int
        转折点之间的最小时间距离
    t_max : int
        转折点之间的最大时间距离（超过t_max即视为满足转折条件）
    min_pct : float
        在满足t_min参数设置的条件下，极大值点和上一个极小值点的最小变化百分比
    min_val : float
        在满足t_min参数设置的条件下，极大值点和上一个极小值点的最小变化绝对值
        （若min_pct设置，则此参数失效）
    max_pct : float
        在不满足t_min参数设置的条件下，极大值点和上一个极小值点的变化百分比若超过此参数值，则视为满足转折条件
    max_val : float
        在不满足t_min参数设置的条件下，极大值点和上一个极小值点的变化绝对值若超过此参数值，则视为满足转折条件
        （若max_pct设置，则此参数失效）
    t_min_up : int
        同t_min，只控制上涨
    t_min_down : int
        同t_min，只控制下跌
    t_max_up : int
        同t_max，只控制上涨
    t_max_down : int
        同t_max，只控制下跌
    up_min_pct : float
        同min_pct，只控制上涨
    up_min_val : float
        同min_val，只控制上涨
    up_max_pct : float
        同max_pct，只控制上涨
    up_max_val : float
        同max_val，只控制上涨
    down_min_pct : float
        同min_pct，只控制下跌
    down_min_val : float
        同min_val，只控制下跌
    down_max_pct : float
        同max_pct，只控制下跌
    down_max_val : float
        同max_val，只控制下跌
    pct_v00 : float
        计算百分比时分母为0指定结果
    logger : logging.Logger
        日志记录器

    Returns
    -------
    zigzag : pd.Series
        返回zigzag标签序列。其中1/-1表示确定的高/低点；
        0.5/-0.5表示未达到偏离条件而不能确定的高低点。
    '''
    
    # 参数转化和检查
    t_min_up = t_min if isnull(t_min_up) else t_min_up
    t_min_down = t_min if isnull(t_min_down) else t_min_down
    t_max_up = t_max if isnull(t_max_up) else t_max_up
    t_max_down = t_max if isnull(t_max_down) else t_max_down
    up_min_pct = min_pct if isnull(up_min_pct) else up_min_pct
    up_min_val = min_val if isnull(up_min_val) else up_min_val
    up_max_pct = max_pct if isnull(up_max_pct) else up_max_pct
    up_max_val = max_val if isnull(up_max_val) else up_max_val
    down_min_pct = -1*min_pct if (isnull(down_min_pct) and not isnull(min_pct)) else down_min_pct
    down_min_val = -1*min_val if (isnull(down_min_val) and not isnull(min_val)) else down_min_val
    down_max_pct = -1*max_pct if (isnull(down_max_pct) and not isnull(max_pct)) else down_max_pct
    down_max_val = -1*max_val if (isnull(down_max_val) and not isnull(max_val)) else down_max_val
    # 上涨-长时最小幅度参数
    assert not (isnull(up_min_pct) and isnull(up_min_val)), \
        '必须设置`up_min_pct`或`up_min_val`, 若要忽略此参数，可将其设置为负数'
    if not isnull(up_min_pct) and not isnull(up_min_val):
        logger_show('同时设置`up_min_pct`和`up_min_val`，以`up_min_pct`为准！',
                    logger, 'warn')
        up_min_val = None
    # 上涨-短时最大幅度参数
    if not isnull(up_max_pct) and not isnull(up_max_val):
        logger_show('同时设置`up_max_pct`和`up_max_val`，以`up_max_pct`为准！',
                    logger, 'warn')
        up_max_val = None
    # 下跌-长时最小幅度参数
    assert not (isnull(down_min_pct) and isnull(down_min_val)), \
        '必须设置`down_min_pct`或`down_min_val`, 若要忽略此参数，可将其设置为正数'
    if not isnull(down_min_pct) and not isnull(down_min_val):
        logger_show('同时设置`down_min_pct`和`down_min_val`，以`down_min_pct`为准！',
                    logger, 'warn')
        down_min_val = None
    # 下跌-短时最大幅度参数
    if not isnull(down_max_pct) and not isnull(down_max_val):
        logger_show('同时设置`down_max_pct`和`down_max_val`，以`down_max_pct`为准！',
                    logger, 'warn')
        down_max_val = None
        
    def _cal_up_dif_min(v0, v1):
        if not isnull(up_min_pct):
            return cal_pct(v0, v1, pct_v00)
        return v1 - v0
    
    def _cal_up_dif_max(v0, v1):
        if not isnull(up_max_pct):
            return cal_pct(v0, v1, pct_v00)
        else:
            if not isnull(up_max_val):
                return v1 - v0
            else:
                return None
    
    def _up_sure(up_dif_min, up_dif_max, t):
        cond1_ = True if isnull(t_min_up) else (t > t_min_up)
        if not isnull(up_min_pct):
            cond1 = (up_dif_min >= up_min_pct) and cond1_
        else:
            cond1 = (up_dif_min >= up_min_val) and cond1_
        if isnull(t_max_up):
            cond2 = False
        else:
            cond2 = (t >= t_max_up)
        cond3_ = True if isnull(t_min_up) else (t <= t_min_up)
        if isnull(up_dif_max):
            cond3 = False
        elif not isnull(up_max_pct):
            cond3 = (up_dif_max >= up_max_pct) and cond3_
        else:
            cond3 = (up_dif_max >= up_max_val) and cond3_
        return (cond1 or cond2 or cond3)
    
    def _cal_down_dif_min(v0, v1):
        if not isnull(down_min_pct):
            return cal_pct(v0, v1, pct_v00)
        return v1 - v0
    
    def _cal_down_dif_max(v0, v1):
        if not isnull(down_max_pct):
            return cal_pct(v0, v1, pct_v00)
        else:
            if not isnull(down_max_val):
                return v1 - v0
            else:
                return None
    
    def _down_sure(down_dif_min, down_dif_max, t):
        cond1_ = True if isnull(t_min_down) else (t > t_min_down)
        if not isnull(down_min_pct):
            cond1 = (down_dif_min <= down_min_pct) and cond1_
        else:
            cond1 = (down_dif_min <= down_min_val) and cond1_
        if isnull(t_max_up):
            cond2 = False
        else:
            cond2 = (t >= t_max_up)
        cond3_ = True if isnull(t_min_down) else (t <= t_min_down)
        if isnull(down_dif_max):
            cond3 = False
        elif not isnull(down_max_pct):
            cond3 = (down_dif_max <= down_max_pct) and cond3_
        else:
            cond3 = (down_dif_max <= down_max_val) and cond3_
        return (cond1 or cond2 or cond3)
    
    def _get_up_dif_max_init(init_val):
        if isnull(up_max_pct) and isnull(up_max_val):
            return None
        return init_val
    
    def _get_down_dif_max_init(init_val):
        if isnull(down_max_pct) and isnull(down_max_val):
            return None
        return init_val

    def confirm_high(df, k):
        '''
        | 从前一个低点位置k开始确定下一个高点位置
        | 注：传入的df必须是reset_index的
        '''
        k0 = k
        v0 = df.loc[k, low_col]
        
        up_dif_min, t_up = -np.inf, 0
        up_dif_max = _get_up_dif_max_init(-np.inf)
        updown_dif_min, t_updown = np.inf, 0
        updown_dif_max = _get_down_dif_max_init(np.inf)
        up_sure, updown_sure = False, False
        
        cummax, cummax_idx = -np.inf, k+1
        k += 2
        while k < df.shape[0] and (not up_sure or not updown_sure):
            if df.loc[k, low_col] < v0:
                cummax_idx = df[high_col].iloc[k0+1:k+1].idxmax()
                return cummax_idx, True, True
            
            if df.loc[k-1, high_col] > cummax:
                cummax = df.loc[k-1, high_col]
                cummax_idx = k-1
                updown_dif_min = np.inf

                up_dif_min = _cal_up_dif_min(v0, cummax)
                up_dif_max = _cal_up_dif_max(v0, cummax)
                t_up = cummax_idx - k0
            
            updown_dif_min_ = _cal_down_dif_min(cummax, df.loc[k, low_col])
            if updown_dif_min_ < updown_dif_min:
                updown_dif_min = updown_dif_min_
                updown_dif_max = _cal_down_dif_max(cummax, df.loc[k, low_col])
                t_updown = k - cummax_idx
                
            up_sure = _up_sure(up_dif_min, up_dif_max, t_up)
            updown_sure = _down_sure(updown_dif_min, updown_dif_max, t_updown)

            k += 1

        if k == df.shape[0]:
            if df.loc[k-1, high_col] > cummax:
                cummax = df.loc[k-1, high_col]
                cummax_idx = k-1
                
                up_dif_min = _cal_up_dif_min(v0, cummax)
                up_dif_max = _cal_up_dif_max(v0, cummax)
                t_up = cummax_idx - k0
                
                updown_dif_min, t_updown = 0.0, 0
                updown_dif_max = _get_down_dif_max_init(0.0)
                
                up_sure = _up_sure(up_dif_min, up_dif_max, t_up)
                updown_sure = _down_sure(updown_dif_min, updown_dif_max, t_updown)
        
        return cummax_idx, up_sure, updown_sure

    def confirm_low(df, k):
        '''
        | 从前一个高点位置k开始确定下一个低点位置
        | 注：传入的df必须是reset_index的
        '''
        k0 = k
        v0 = df.loc[k, high_col]

        down_dif_min, t_down = np.inf, 0
        down_dif_max = _get_down_dif_max_init(np.inf)
        downup_dif_min, t_downup = -np.inf, 0
        downup_dif_max = _get_up_dif_max_init(-np.inf)
        down_sure, downup_sure = False, False

        cummin, cummin_idx = np.inf, k+1
        k += 2
        while k < df.shape[0] and (not down_sure or not downup_sure):
            if df.loc[k, high_col] > v0:
                cummin_idx = df[low_col].iloc[k0+1:k+1].idxmin()
                return cummin_idx, True, True
            
            if df.loc[k-1, low_col] < cummin:
                cummin = df.loc[k-1, low_col]
                cummin_idx = k-1
                downup_dif_min = -np.inf

                down_dif_min = _cal_down_dif_min(v0, cummin)
                down_dif_max = _cal_down_dif_max(v0, cummin)
                t_down = cummin_idx - k0
            
            downup_dif_min_ = _cal_up_dif_min(cummin, df.loc[k, high_col])
            if downup_dif_min_ > downup_dif_min:
                downup_dif_min = downup_dif_min_
                downup_dif_max = _cal_up_dif_max(cummin, df.loc[k, high_col])
                t_downup = k - cummin_idx
                
            down_sure = _down_sure(down_dif_min, down_dif_max, t_down)
            downup_sure = _up_sure(downup_dif_min, downup_dif_max, t_downup)
            
            k += 1

        if k == df.shape[0]:
            if df.loc[k-1, low_col] < cummin:
                cummin = df.loc[k-1, low_col]
                cummin_idx = k-1
                
                down_dif_min = _cal_down_dif_min(v0, cummin)
                down_dif_max = _cal_down_dif_max(v0, cummin)
                t_down = cummin_idx - k0
                
                downup_dif_min, t_downup = 0.0, 0
                downup_dif_max = _get_up_dif_max_init(0.0)
                
                down_sure = _down_sure(down_dif_min, down_dif_max, t_down)
                downup_sure = _up_sure(downup_dif_min, downup_dif_max, t_downup)

        return cummin_idx, down_sure, downup_sure
    
    def zigzag_from_k(df, k, ktype):
        '''
        | 从已经确认的转折点k处往后计算所有转折点
        | 注：传入的df必须是reset_index的
        '''
        assert ktype in [1, -1]
        df = df.copy()
        while k < df.shape[0]:
            func_confirm = confirm_high if ktype == -1 else confirm_low
            k, ok_mid, ok_right = func_confirm(df, k)
            if ok_mid and ok_right:
                df.loc[k, 'zigzag'] = -ktype
                ktype = -ktype
            elif ok_mid and not ok_right:
                df.loc[k, 'zigzag'] = -ktype * 0.5
                break
            # elif not ok_mid and ok_right:
            #     df.loc[k, 'zigzag'] = -ktype
            #     ktype = -ktype
            #     break
            else:
                break
        return df['zigzag']
    
    # 无效值检查
    cols = list(set([high_col, low_col]))
    for c in cols:
        assert df[c].isna().sum() == 0, '检测到无效值，请检查数据！'
        assert (df[c] == np.inf).sum() == 0, '检测到无穷大值，请检查数据！'
        assert (df[c] == -np.inf).sum() == 0, '检测到负无穷大值，请检查数据！'
    
    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df = df.reset_index(drop=True)

    # 若data中已有zigzag列，先检查找出最后一个转折点已经能确定的位置，从此位置开始算
    if 'zigzag' in df.columns:
        cols = cols + ['zigzag']
        df = df[cols].copy()
        k = df.shape[0] - 1
        while k > 0 and df.loc[k, 'zigzag'] in [0, 0.5, -0.5]:
            k -= 1
        ktype = df.loc[k, 'zigzag']
        if ktype in [1, -1]:
            df['zigzag'] = zigzag_from_k(df, k, ktype)
            
    # 若data中没有zigzag列或已有zigzag列不能确定有效的转折点，则需要全部计算
    if 'zigzag' not in df.columns or ktype in [0, 0.5, -0.5]:
        df = df[cols].copy()
        df['zigzag'] = 0
        
        # 先找到序列最高点和最低点的位置，从序号较大者往前算出高低点
        imax, imin = df[high_col].idxmax(), df[low_col].idxmin()
        if imax > imin:
            icut, ktype0 = imax, 1
        else:
            icut, ktype0 = imin, -1
        df0 = df.iloc[:icut+1, :].copy()[::-1].reset_index(drop=True)
        df0['zigzag'] = zigzag_from_k(df0, 0, ktype0)
        df0 = df0[::-1].reset_index(drop=True)
        
        # TODO: 确定方式二是否完全准确
        # 方式一：先确定第一个高/低点标签，再往后计算
        k, ktype = get_first_appear(df0['zigzag'], lambda x: x in [-1, 1])
        if not isnull(k):
            df['zigzag'] = df0['zigzag'].iloc[:k+1]
            df['zigzag'] = df['zigzag'].fillna(0)
            df['zigzag'] = zigzag_from_k(df, k, ktype)
        # # 方式二：直接使用df0中的结果
        # df['zigzag'] = df0['zigzag']
        # df['zigzag'] = df['zigzag'].fillna(0)
        # k, ktype = get_first_appear(df0['zigzag'],
        #                             lambda x: x in [-1, 1],
        #                             reverse=True)
        # if not isnull(k):
        #     df['zigzag'] = zigzag_from_k(df, k, ktype)
        
    df.index = ori_index

    return df['zigzag']


if __name__ == '__main__':
    import numpy as np
    from dramkit.datsci import find_maxmin
    from dramkit import TimeRecoder, plot_series, load_csv
    from finfactory.finplot.plot_candle import plot_candle
    
    def plot_candle_zz(data, zzcol='zigzag',
                       zz_high='high', zz_low='low',
                       **kwargs):
        '''在K线图上绘制ZigZag'''
        # data = data.copy()
        data['col_zz1'] = data[zzcol].apply(lambda x: 1 if x > 0 else 0)
        data['col_zz-1'] = data[zzcol].apply(lambda x: 1 if x < 0 else 0)
        data['col_zz'] = data['col_zz1'] * data[zz_high] + \
                         data['col_zz-1'] * data[zz_low]
        data['col_zz'] = data[['col_zz1', 'col_zz-1', 'col_zz']].apply(
                         lambda x: x['col_zz'] if x['col_zz1'] == 1 or \
                                   x['col_zz-1'] == 1 else np.nan, axis=1)
        data['zz_loc'] = data['col_zz1'] + data['col_zz-1']
        if 'cols_to_label_info' in kwargs.keys():
            cols_to_label_info = kwargs['cols_to_label_info']
            del kwargs['cols_to_label_info']
            cols_to_label_info.update(
                {'col_zz': [['zz_loc', (1,), ('-b',), False]]})
        else:
            cols_to_label_info={'col_zz': [['zz_loc', (1,), ('-b',), False]]}
        if 'cols_other_upleft' in kwargs.keys():
            cols_other_upleft = kwargs['cols_other_upleft']
            del kwargs['cols_other_upleft']
            cols_other_upleft.update({'col_zz': ('.b', False)})
        else:
            cols_other_upleft = {'col_zz': ('.w', False)}
        plot_candle(data, cols_other_upleft=cols_other_upleft,
                    cols_to_label_info=cols_to_label_info,
                    **kwargs)
    
    
    tr = TimeRecoder()
    
    # '''
    # 二次曲线叠加正弦余弦
    N = 200
    t = np.linspace(0, 1, N)
    s = 6*t*t + np.cos(10*2*np.pi*t*t) + np.sin(6*2*np.pi*t)
    df = pd.DataFrame(s, columns=['test'])

    # t_min = None
    t_min = 5
    df['label'] = find_maxmin(df['test'], t_min=t_min, min_dif_val=4, t_max=10)
    plot_series(df, {'test': ('.-k', None)},
                cols_to_label_info={
                    'test': [
                        ['label', (-1, 1), ('r^', 'gv'), None]
                        ]},
                figsize=(8, 3))
    df['zigzag'] = zigzag(df, 'test', 'test',
                          t_min_up=t_min, t_min_down=t_min,
                          t_max_up=10, t_max_down=10,
                          up_min_val=4, down_min_val=-4,
                          up_min_pct=None, down_min_pct=None
                          )
    df['zigzag'] = df['zigzag'].replace({0.5: 1, -0.5: -1})
    plot_series(df, {'test': ('.-k', None)},
                cols_to_label_info={
                    'test': [
                        ['zigzag', (-1, 1), ('r^', 'gv'), None]
                        ]},
                figsize=(8, 3))
    # '''


    # '''
    fpath = '../_test/510050.SH_daily_qfq.csv'
    his_data = load_csv(fpath)
    his_data.rename(columns={'date': 'time'}, inplace=True)
    his_data.set_index('time', drop=False, inplace=True)

    # N = his_data.shape[0]
    N = 100
    data = his_data.iloc[-N:-1, :].copy()

    high_col, low_col = 'high', 'low'
    params = {'high_col': high_col,
              'low_col': low_col,
              'up_min_pct': 3/100,
              'down_min_pct': -3/100
              }
    data['zigzag'] = zigzag(data, **params)
    plot_candle_zz(data, zz_high=high_col, zz_low=low_col,
                   args_ma=None, args_boll=None, plot_below=None,
                   grid=False, figsize=(10, 6))
    # '''


    # '''
    fpath = '../_test/zigzag_test2.csv'
    df = load_csv(fpath)
    df = df[['date', 'time', 'open', 'high', 'low', 'close', 'label']]
    dates = list(df['date'].unique())
    df = df[df['date'] == dates[0]].copy()
    plot_candle_zz(df, zzcol='label',
                   args_ma=None,
                   args_boll=None,
                   plot_below=None,
                   figsize=(10, 6))
    
    high_col, low_col = 'high', 'low'
    min_pct = -1/100
    t_min = 15
    params = {'high_col': high_col,
              'low_col': low_col,
              'min_pct': min_pct,
              't_min': t_min
              }    
    
    df['zigzag'] = zigzag(df, **params)
    plot_candle_zz(df,
                   args_ma=None,
                   # args_boll=[15, 2],
                   args_boll=None,
                   plot_below=None,
                   figsize=(10, 6))
    
    # '''
    
    tr.used()
