# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections.abc import Callable

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dramkit.gentools import con_count, isnull


def cut_mad(series, n=3,
            up_cut_val=None,
            low_cut_val=None,
            func_filter=None):
    '''
    | MAD极端值处理方法
    | func_filter为对计算中位数和偏差之前对数据进行筛的函数，返回值为True的才参与计算
    '''
    isinstance(func_filter, (Callable, type(None)))
    values = np.array(series)
    if isnull(func_filter):
        values_ = values.copy()
    else:
        values_ = values[func_filter(values)]
    median = np.nanmedian(values_, axis=0)
    abs_med_dif = np.abs(values_ - median)
    med = np.nanmedian(abs_med_dif) * 1.483
    upper = median + n * med
    lower = median - n * med
    if isnull(up_cut_val):
        up_cut_val = upper
    if isnull(low_cut_val):
        low_cut_val = lower
    values[values > upper] = up_cut_val
    values[values < lower] = low_cut_val
    try:
        return pd.Series(values, index=series.index)
    except:
        return values
    
    
def cut_nstd(series, n=3, up_cut_val=None, low_cut_val=None):
    '''标准差倍数截断方法，极端值处理'''
    df = pd.DataFrame({'s': series})
    mean = df['s'].mean()
    std = df['s'].std()
    upper = mean + n * std
    lower = mean - n * std
    if isnull(up_cut_val):
        up_cut_val = upper
    if isnull(low_cut_val):
        low_cut_val = lower
    df.loc[df['s'] > upper, 's'] = up_cut_val
    df.loc[df['s'] < lower, 's'] = low_cut_val
    return df['s']


class ExternalStd(object):
    '''极端值处理，标准差倍数法'''

    def __init__(self, nstd=3, cols=None):
        raise NotImplementedError

    # def deal_ext_value(df_fit, cols=None, dfs_trans=None, nstd=5):
    #     '''极端值处理'''

    #     cols = df_fit.columns if cols is None else cols

    #     mean_stds = []
    #     for col in cols:
    #         mean_stds.append((col, df_fit[col].mean(), df_fit[col].std()))

    #     df_fited = df_fit.copy()
    #     for (col, mean, std) in mean_stds:
    #         df_fited[col] = df_fited[col].apply(
    #                         lambda x: np.clip(x, mean-nstd*std, mean+nstd*std))

    #     if dfs_trans is not None:
    #         dfs_traned = []
    #         for df in dfs_trans:
    #             df_tmp = df.copy()
    #             for (col, mean, std) in mean_stds:
    #                 df_tmp[col] = df_tmp[col].apply(
    #                         lambda x: np.clip(x, mean-nstd*std, mean+nstd*std))
    #             dfs_traned.append(df_tmp)

    #     return df_fited, tuple(dfs_traned)


def norm_std(series, reverse=False, ddof=1, return_mean_std=False):
    '''
    z-score标准化

    Parameters
    ----------
    series : series, pd.Series, np.array
        待标准化序列数据
    reverse : 
        是否反向标准化
    ddof : int
        | 指定计算标准差时是无偏还是有偏的：
        | ddof=1时计算的是无偏标准差（样本标准差，分母为n-1），
        | ddof=0时计算的是有偏标准差（总体标准差，分母为n）

        .. hint::
            | pandas的std()默认计算的是无偏标准差，
            | numpy的std()默认计算的是有偏标准差
    return_mean_std : bool
        为True时同时返回均值和标准差，为False时不返回

    Note
    ----
    | Z-score适用于series的最大值和最小值未知或有超出取值范围的离群值的情况。
    | (一般要求原始数据的分布可以近似为高斯分布，否则效果可能会很糟糕)

    References
    ----------
    https://blog.csdn.net/qxqxqzzz/article/details/88663198
    '''
    Smean, Sstd = series.mean(), series.std(ddof=ddof)
    if not reverse:
        series_new = (series - Smean) / Sstd
    else:
        series_new = (Smean - series) / Sstd
    if not return_mean_std:
        return series_new
    else:
        return series_new, (Smean, Sstd)


def norm_linear(x, xmin, xmax, ymin=0, ymax=1, reverse=False,
                x_must_in_range=True, v_xmin_eq_xmax=np.nan,
                v_ymin_eq_ymax=np.nan):
    '''
    | 线性映射：
    | 将取值范围在[xmin, xmax]内的x映射到取值范围在[ymin, ymax]内的xnew
    | reverse设置是否反向，若为True，则映射到[ymax, ymin]范围内
    | x_must_in_range设置当x不在[xmin, xmax]范围内时是否报错
    | v_xmin_eq_xmax设置当xmin和xmax相等时的返回值
    '''

    if x_must_in_range:
        if x < xmin or x > xmax:
            raise ValueError('必须满足 xmin =< x <= xmax ！')

    if xmin > xmax or ymin > ymax:
        raise ValueError('必须满足 xmin <= xmax 且 ymin <= ymax ！')

    if xmin == xmax:
        return v_xmin_eq_xmax

    if ymin == ymax:
        return v_ymin_eq_ymax

    if reverse:
        ymin, ymax = ymax, ymin

    xnew = ymin + (x-xmin) * (ymax-ymin) / (xmax-xmin)

    return xnew


def series_norm_linear(series, ymin=0, ymax=1,
                       reverse=False):
    '''对series的值进行线性映射，参考 :func:`norm_linear` 函数'''
    x = pd.Series(series)
    xmin, xmax = x.min(), x.max()
    if reverse:
        ymin, ymax = ymax, ymin
    res = ymin + (x-xmin) * (ymax-ymin) / (xmax-xmin)
    return res


def norm_mid(x, xmin, xmax, ymin=0, ymax=1, xbest=None):
    '''
    | 中间型（倒V型）指标的正向化线性映射，新值范围为[nmin, ymax]
    | (指标值既不要太大也不要太小，适当取中间值最好，如水质量评估PH值)
    | xmin和xmax为指标可能取到的最小值和最大值
    | xbest指定最优值，若不指定则将xmin和xmax的均值当成最优值
    | 参考：https://zhuanlan.zhihu.com/p/37738503
    '''
    xbest = (xmin+xmax)/2 if xbest is None else xbest
    if x <= xmin or x >= xmax:
        return ymin
    elif x > xmin and x < xbest:
        return norm_linear(x, xmin, xbest, ymin, ymax)
    elif x < xmax and x >= xbest:
        return norm_linear(x, xbest, xmax, ymin, ymax, reverse=True)


def norm01_mid(x, xmin, xmax, xbest=None):
    '''
    | 中间型（倒V型）指标的正向化和（线性）01标准化
    | (指标值既不要太大也不要太小，适当取中间值最好，如水质量评估PH值)
    | xmin和xmax为指标可能取到的最小值和最大值
    | xbest指定最优值，若不指定则将xmin和xmax的均值当成最优值
    | 参考：https://zhuanlan.zhihu.com/p/37738503
    '''
    xbest = (xmin+xmax)/2 if xbest is None else xbest
    if x <= xmin or x >= xmax:
        return 0
    elif x > xmin and x < xbest:
        return (x - xmin) / (xbest - xmin)
    elif x < xmax and x >= xbest:
        return (xmax - x) / (xmax - xbest)


def norm_side(x, xmin, xmax, ymin=0, ymax=1, xworst=None, v_out_limit=None):
    '''
    | 两边型（V型）指标的正向化线性映射，新值范围为[ymin, ymax]
    | (指标越靠近xmin或越靠近xmax越好，越在中间越不好)
    | xmin和xmax为指标可能取到的最小值和最大值
    | xworst指定最差值，若不指定则将xmin和xmax的均值当成最差值
    | v_out_limit指定当x超过xmin或xmax界限之后的正向标准化值，不指定则默认为ymax
    '''
    v_out_limit = ymax if v_out_limit is None else v_out_limit
    xworst = (xmin+xmax)/2 if xworst is None else xworst
    if x < xmin or x > xmax:
        return v_out_limit
    elif x >= xmin and x < xworst:
        return norm_linear(x, xmin, xworst, ymin, ymax, reverse=True)
    elif x <= xmax and x >= xworst:
        return norm_linear(x, xworst, xmax, ymin, ymax)


def norm01_side(x, xmin, xmax, xworst=None, v_out_limit=1):
    '''
    | 两边型（V型）指标的正向化和（线性）01标准化
    | (指标越靠近xmin或越靠近xmax越好，越在中间越不好)
    | xmin和xmax为指标可能取到的最小值和最大值
    | xworst指定最差值，若不指定则将xmin和xmax的均值当成最差值
    | v_out_limit指定当x超过xmin或xmax界限之后的正向标准化值，不指定则默认为1
    '''
    xworst = (xmin+xmax)/2 if xworst is None else xworst
    if x < xmin or x > xmax:
        return v_out_limit
    elif x >= xmin and x < xworst:
        return (xworst - x) / (xworst - xmin)
    elif x <= xmax and x >= xworst:
        return (x - xworst) / (xmax - xworst)


def norm_range(x, xmin, xmax, xminmin, xmaxmax, ymin=0, ymax=1):
    '''
    | 区间型指标的正向化正向化线性映射，新值范围为[ymin, ymax]
    | (指标的取值最好落在某一个确定的区间最好，如体温)
    | [xmin, xmax]指定指标的最佳稳定区间，[xminmin, xmaxmax]指定指标的最大容忍区间
    | 参考：https://zhuanlan.zhihu.com/p/37738503
    '''
    if x >= xmin and x <= xmax:
        return ymax
    elif x <= xminmin or x >= xmaxmax:
        return ymin
    elif x > xmax and x < xmaxmax:
        return norm_linear(x, xmax, xmaxmax, ymin, ymax, reverse=True)
    elif x < xmin and x > xminmin:
        return norm_linear(x, xminmin, xmin, ymin, ymax)


def norm01_range(x, xmin, xmax, xminmin, xmaxmax):
    '''
    | 区间型指标的正向化和（线性）01标准化
    | (指标的取值最好落在某一个确定的区间最好，如体温)
    | [xmin, xmax]指定指标的最佳稳定区间，[xminmin, xmaxmax]指定指标的最大容忍区间
    | 参考：https://zhuanlan.zhihu.com/p/37738503
    '''
    if x >= xmin and x <= xmax:
        return 1
    elif x <= xminmin or x >= xmaxmax:
        return 0
    elif x > xmax and x < xmaxmax:
        return 1 - (x - xmax) / (xmaxmax - xmax)
    elif x < xmin and x > xminmin:
        return 1 - (xmin - x) / (xmin - xminmin)


def get_pca(df, dfs_trans=None, **kwargs_pca):
    '''
    PCA主成分转化

    Parameters
    ----------
    df : pandas.DataFrame
        用于训练PCA模型的数据
    df_trans : None, list
        | 待转化数据列表，每个元素都是pandas.DataFrame，
        | 即用训练好的PCA模型对dfs_trans中所有的数据进行主成分转化
    **kwargs_pca :
        sklearn.decomposition.PCA接受的参数

    Returns
    -------
    df_pca : pandas.DataFrame
        df进行PCA转化之后的数据，列名格式为'imfk'(第k列表示第k个主成分)
    dfs_transed : tuple
        利用df训练好的PCA模型对dfs_trans中的数据进行主成分转化之后的结果

    TODO
    ----
    改成跟 :func:`scale_skl` 和 :func:`scale_skl_inverse` 函数那样
    有返回训练好的模型以及用训练好的模型进行转换
    '''
    mdl = PCA(**kwargs_pca)
    mdl.fit(df)
    def pca_trans(df):
        df_ = mdl.transform(df)
        df_ = pd.DataFrame(df_)
        df_.columns = ['imf'+str(_) for _ in range(1, df_.shape[1]+1)]
        return df_
    df_pca = pca_trans(df)
    if dfs_trans is not None:
        dfs_transed = []
        for df_ in dfs_trans:
            dfs_transed.append(df_)
    else:
        dfs_transed = None
    return df_pca, tuple(dfs_transed)


def scale_skl(df_fit, dfs_trans=None, cols=None, scale_type='std', **kwargs):
    '''
    用sklearn进行数据标准化处理

    Parameters
    ----------
    df_fit : pandas.DataFrame
        训练数据，用于计算标准化中间变量
    dfs_trans : None, pandas.DataFrame, list
        需要以df_fit为基准进行标准化的df或df列表
    cols : None, list
        指定需要标准化的列，默认对所有列进行标准化
    scale_type : str
        | 指定标准化类型:
        | ``std`` 或 ``z-score`` 使用 ``sklearn.preprocessing.StandardScaler``
        | ``maxmin`` 或 ``minmax`` 使用 ``sklearn.preprocessing.MinMaxScaler``
    **kwargs :
        接收对应Scaler支持的参数

    Returns
    -------    
    df_fited : pandas.DataFrame
        df_fit标准化之后的数据
    dfs_transed : None, pandas.DataFrame, tuple
        dfs_trans标准化之后对应的数据
    scaler_info : tuple
        包含scaler即df_fit计算得到的标准化中间变量信息; cols即用到的列名列表
    '''

    sklScaler_map = {'std': StandardScaler, 'z-score': StandardScaler,
                     'maxmin': MinMaxScaler, 'minmax': MinMaxScaler}
    sklScaler = sklScaler_map[scale_type]

    cols_all = list(df_fit.columns)

    if cols is None:
        scaler = sklScaler(**kwargs).fit(df_fit)
        df_fited = pd.DataFrame(scaler.transform(df_fit),
                                         columns=cols_all, index=df_fit.index)
        if dfs_trans is None:
            return df_fited, None, (scaler, cols_all)
        if isinstance(dfs_trans, pd.core.frame.DataFrame):
            dfs_transed = pd.DataFrame(scaler.transform(dfs_trans),
                                       columns=cols_all,
                                       index=dfs_trans.index)
        else:
            dfs_transed = [pd.DataFrame(scaler.transform(df), columns=cols_all,
                           index=df.index) for df in dfs_trans]
        return df_fited, dfs_transed, (scaler, df_fit.columns)

    cols_rest = [x for x in cols_all if x not in cols]
    df_toFit = df_fit.reindex(columns=cols)

    scaler = sklScaler(**kwargs).fit(df_toFit)
    df_fited = pd.DataFrame(scaler.transform(df_toFit), columns=cols,
                                                        index=df_toFit.index)
    for col in cols_rest:
        df_fited[col] = df_fit[col]
    df_fited = df_fited.reindex(columns=cols_all)
    if dfs_trans is None:
        return df_fited, None, (scaler, cols)

    dfs_transed = []
    if isinstance(dfs_trans, pd.core.frame.DataFrame):
        dfs_trans_ = [dfs_trans]
    else:
        dfs_trans_ = dfs_trans
    for df in dfs_trans_:
        df_trans = df.reindex(columns=cols)
        df_transed = pd.DataFrame(scaler.transform(df_trans),
                                  columns=cols, index=df_trans.index)
        cols_all_df = list(df.columns)
        cols_rest_df = [x for x in cols_all_df if x not in cols]
        for col in cols_rest_df:
            df_transed[col] = df[col]
        dfs_transed.append(df_transed.reindex(columns=cols_all_df))
    if isinstance(dfs_trans, pd.core.frame.DataFrame):
        dfs_transed = dfs_transed[0]
    else:
        dfs_transed = tuple(dfs_transed)
    return df_fited, dfs_transed, (scaler, cols)


def scale_skl_inverse(scaler, dfs_to_inv, cols=None, **kwargs):
    '''
    反标准化还原数据

    Parameters
    ----------
    scaler : class
        fit过的sklearn Scaler类(如StandardScaler、MinMaxScaler)
    dfs_to_inv : pandas.DataFrame, list
        待还原的df或df列表
    cols : None, list
        指定需要还原的列名列表，默认所有列
    **kwargs :
        接收scaler.inverse_transform函数支持的参数


    :returns: `pandas.DataFrame, tuple` - 反标准化还原之后对应的数据

    Note
    ----
    注：(scaler, cols)应与scale_skl函数输出一致
    '''

    dfs_inved = []
    if isinstance(dfs_to_inv, pd.core.frame.DataFrame):
        dfs_to_inv_ = [dfs_to_inv]
    else:
        dfs_to_inv_ = dfs_to_inv

    if cols is None:
        for df in dfs_to_inv_:
            df_inved = pd.DataFrame(scaler.inverse_transform(df, **kwargs),
                                    columns=df.columns, index=df.index)
            dfs_inved.append(df_inved)
        if isinstance(dfs_to_inv, pd.core.frame.DataFrame):
            dfs_inved = dfs_inved[0]
        else:
            dfs_inved = tuple(dfs_inved)
        return dfs_inved

    for df in dfs_to_inv_:
        cols_all = list(df.columns)
        cols_rest = [x for x in cols_all if x not in cols]
        df_inved = pd.DataFrame(scaler.inverse_transform(df[cols], **kwargs),
                                columns=cols, index=df.index)
        for col in cols_rest:
            df_inved[col] = df[col]
        dfs_inved.append(df_inved.reindex(columns=cols_all))

    if isinstance(dfs_to_inv, pd.core.frame.DataFrame):
        dfs_inved = dfs_inved[0]
    else:
        dfs_inved = tuple(dfs_inved)
    return dfs_inved


def get_miss_rate(df, cols=None, return_type='dict'):
    '''
    计算df(`pd.DataFrame`)中指定cols列的缺失率，return_type可选['dict', 'df']
    
    TODO
    ----
    cols可为str，指定单列，返回单个值
    '''
    assert cols is None or isinstance(cols, list)
    if isinstance(cols, list):
        for col in cols:
            if col not in df.columns:
                raise ValueError('{}不在df列中'%col)
    df = df.copy() if cols is None else df.reindex(columns=cols)
    mis_rates = df.isnull().sum() / df.shape[0]
    if return_type == 'dict':
        return mis_rates.to_dict()
    elif return_type == 'df':
        mis_rates = pd.DataFrame(mis_rates).reset_index()
        mis_rates.columns = ['col', 'miss_pct']
        mis_rates.sort_values('miss_pct', ascending=False, inplace=True)
        return mis_rates
    
    
def drop_miss_feature(data, cols=None, thre=0.8):
    '''删除缺失率大于thre的列'''
    loss_rates = get_miss_rate(data, cols=cols)
    drop_cols = [x for x in loss_rates if loss_rates[x] > thre]
    df = data.drop(drop_cols, axis=1)
    return df


def drop_miss_sample(data, thre=0.5, cols=None):
    '''
    | 删除数据缺失率大于thre的样本（行）
    | 注：应保证data.index是唯一的
    '''
    assert isinstance(cols, (type(None), list))
    if isnull(cols):
        cols = list(data.columns)
    df = data.copy()
    df['loss_rate'] = df[cols].isna().sum(axis=1) / df.shape[1]
    df = df[df['loss_rate'] <= thre]
    df = df.drop('loss_rate', axis=1)
    return df


def fillna_ma(series, ma=None, ma_min=2):
    '''
    | 用移动平均MA填充序列series(`pd.Series`)中的缺失值
    | ma(`int`)设置填充时向前取平均数用的期数，ma_min(`int`)设置最小期数
    | 若ma为None，则根据最大连续缺失记录数确定ma期数(取2*最大连续缺失期数)
    | 返回替换之后的 `pd.Series`
    '''

    if series.name is None:
        series.name = 'series'
    col = series.name
    df = pd.DataFrame(series)

    if ma is None:
        tmp = con_count(series, lambda x: True if isnull(x) else False)
        ma = 2 * tmp.max()
        ma = max(ma, ma_min*2)

    df[col+'_ma'] = df[col].rolling(ma, ma_min).mean()

    df[col] = df[[col, col+'_ma']].apply(lambda x:
               x[col] if not isnull(x[col]) else \
               (x[col+'_ma'] if not isnull(x[col+'_ma']) else x[col]), axis=1)

    return df[col]


def fillna_by_mean(df, cols=None):
    '''
    用列均值替换df(`pd.DataFrame`)中的无效值，cols(`list`)指定待替换列
    '''
    assert cols is None or isinstance(cols, list)
    df = df.copy()
    if cols is None:
        cols = list(df.columns)
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())
    return df


def fillna_by_median(df, cols=None):
    '''
    用列中位数替换df(`pd.DataFrame`)中的无效值，cols(`list`)指定待替换列
    '''
    assert cols is None or isinstance(cols, list)
    df = df.copy()
    if cols is None:
        cols = list(df.columns)
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    return df


if __name__ == '__main__':
    fpath = './_test/No.60_CART_breakup_predict_data.csv'
    df = pd.read_csv(fpath)

    mis_rates = get_miss_rate(df)
    mis_rates = {k: v for k, v in mis_rates.items() if v > 0}

    cols = ['act']
    df_ = fillna_by_mean(df, cols)
    df__ = fillna_by_mean(df, cols)

    a_ = fillna_ma(df['am'])
