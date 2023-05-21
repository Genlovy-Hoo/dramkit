# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.rcParams['font.family'] = ['sans-serif', 'stixgeneral', 'serif']
mpl.rcParams['font.sans-serif'] = ['SimHei', 'KaiTi', 'FangSong']
mpl.rcParams['font.serif'] = ['cmr10', 'SimHei', 'KaiTi', 'FangSong']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm' # 'dejavusans', 'cm', 'stix'
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lr


def plot_scatter(data, colx, coly, reg_type=None, dotstyl='.k',
                 group_col=None, group_dotstyles={}, regstyl='-b',
                 figsize=(10, 7), title=None, xlabel=None,
                 ylabel=None, fontsize=15, nXticks=8,
                 fig_save_path=None):
    '''
    | 散点图绘制
    | group_dotstyles格式: {-1: 'ob', 1: 'or', 0: 'ok'}
    '''
    cols = [colx, coly]
    if not group_col is None:
        cols.append(group_col)
    df = data[list(set(cols))].copy()
    _, ax = plt.subplots(figsize=figsize)
    if group_col is None:
        ax.plot(df[colx], df[coly], dotstyl)
    else:
        for val, dotstyle in group_dotstyles.items():
            dftmp = df[df[group_col] == val]
            ax.plot(dftmp[colx], dftmp[coly], dotstyle,
                    label='%s=%s'%(group_col, val))
    if reg_type in ['lr', 'linear']:
        mdl = lr().fit(df[[colx]], df[coly])
        df['yreg'] = mdl.predict(df[[colx]])
        if mdl.intercept_ > 0:
            lblstr = 'y = {a}x + {b}'.format(
                     a=round(mdl.coef_[0], 4), b=round(mdl.intercept_, 4))
        else:
            lblstr = 'y = {a}x {b}'.format(
                     a=round(mdl.coef_[0], 4), b=round(mdl.intercept_, 4))
        ax.plot(df[colx], df['yreg'], regstyl, label=lblstr)
    plt.legend(loc=0, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    xlabel = colx if xlabel is None else xlabel
    ylabel = coly if ylabel is None else ylabel
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.show()
