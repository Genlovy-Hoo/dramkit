# -*- coding: utf-8 -*-

'''绘制K线图'''

import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set()

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pylab import date2num
from mplfinance.original_flavor import candlestick_ochl
# from ._mpfold import candlestick_ochl
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

from dramkit.logtools.utils_logger import logger_show


def plot_candle(data,
                # args_ma=[5, 10, 20, 30, 60],
                args_ma=None,
                args_boll=[15, 2],
                rects=None,
                cols_other_upleft={}, cols_other_upright={},
                plot_below='volume', args_ma_below=[3, 5, 10],
                cols_other_lowleft={}, cols_other_lowright={},
                cols_to_label_info={}, xparls_info={},
                yparls_info_up=None, yparls_info_low=None,
                ylabels=None, xlabels=None, yscales=None,
                figsize=(11, 7), fig_save_path=None, title=None,
                fontsize_label=15, fontsize_legend=15, nXticks=8,
                width=0.5, alpha=0.95, grid=False, markersize=12,
                logger=None):
    '''
    绘制K线图（蜡烛图）

    .. todo::
        - cols_to_label_info增加标注透明度设置
        - 增加上下图双坐标轴标签、刻度和文本字体格式等设置

    Parameters
    ----------
    data : pandas.DataFrame
        待绘图数据，必须有'time'|'date'、'open'、'high'、'low'、'close'五列，
        以及 ``plot_below`` 参数指定的列名
    args_ma : None, list
        绘制指定移动均线（MA）列表，None时不绘制
    args_boll : None, list
        绘制布林带参数[lag, width]

        .. note::
            args_ma和args_boll必须有一个为None
    rects : list
        | 矩形标注信息，格式为：
        | ``[[(left_low_x, left_low_y), width, height], ...]``
    cols_other_x : dict
        | x可为'upleft', 'upright', 'lowleft', 'lowright'，分别设置在
          上图左轴、上图右轴、下图左轴、下图右轴需要绘制的列信息，格式为：
        | ``{col: (lnstyl, label), ...}``
    cols_to_label_info : dict
        设置需要特殊标注的列绘图信息，格式形如:

        .. code-block:: python

            {col1: [[col_lbl1, (v1, v2, ..), (ln1, ln2, ..), (lbl1, lbl2, ..)],
                    [col_lbl2, (v1, v2, ..), ...]],
             col2: ..}

        其中col是需要被特殊标注的列，col_lbl为标签列；
        v指定哪些标签值对应的数据用于绘图；ln设置线型；
        lbl设置图例标签，若为None，则设置为v，若为False，则不设置图例标签
    xparls_info : dict
        | 设置x轴平行线信息，格式形如：
        | ``{col1: [(yval1, clor1, styl1, width1), (yval2, ...)], col2:, ...}``
        | 其中yval指定平行线y轴位置，clor设置颜色，styl设置线型，width设置线宽
    yparls_info_x : None, list
        | x可为'up', 'low', 分别设置顶部和底部x轴平行线格式信息，格式形如：
        | ``[(xval1, clor1, styl1, width1), (xval2, clor2, style2, width2), ...]``
        | 其中xval指定平行线x轴位置，clor设置颜色，styl设置线型，width设置线宽
    ylabels : None, list
        设置四个y轴标签文本内容，若为None则不设置标签文本，
        若为False则既不设置y轴标签文本内容，也不显示y轴刻度
    xlabels : None, list
        设置两个x轴标签文本内容，若为None则不设置标签文本，
        若为False则既不设置x轴标签文本内容，也不显示x轴刻度
    yscales : None, list
        y轴标轴尺度设置，若为None，则默认普通线性坐标，
        可设置为list指定每个坐标尺度(参见matplotlib中的set_yscale)
    plot_below : None, str
        在K线底部绘制柱状图所用的列名，None时不绘制
    args_ma_below : None, list
        底部图均线（MA）列表，None时不绘制
    width : float
        控制蜡烛宽度
    alpha : float
        控制颜色透明度
    grid : bool
        设置是否显示网格
    
    References
    ----------
    https://github.com/matplotlib/mplfinance
    '''

    # y轴标签设置
    if ylabels is None:
        ylabels = [None, None, None, None]

    # x轴标签设置
    if xlabels is None:
        xlabels = [None, None]

    # 坐标轴尺度设置
    if yscales is None:
        yscales = ['linear'] * 4

    if len(cols_other_lowleft) == 0 and len(cols_other_lowright) > 0:
        logger_show('当底部图只指定右边坐标轴时，默认绘制在左边坐标轴！', logger, 'warn')
        cols_other_lowleft, cols_other_lowright = cols_other_lowright, {}

    if 'time' in data.columns:
        tcol = 'time'
    elif 'date' in data.columns:
        tcol = 'date'
    else:
        raise ValueError('data必须包含`time`或`date`列！')
    if plot_below:
        cols_must = [tcol, 'open', 'high', 'low', 'close'] + [plot_below]
        if not all([x in data.columns for x in cols_must]):
            raise ValueError(
                'data必须包含time, open, high, low, close及plot_below指定的列！')

    cols = [tcol, 'open', 'high', 'low', 'close']
    cols = cols + [plot_below] if plot_below else cols
    if len(cols_other_upleft) > 0:
        cols = cols + list(cols_other_upleft.keys())
    if len(cols_other_upright) > 0:
        cols = cols + list(cols_other_upright.keys())
    if len(cols_other_lowleft) > 0:
        cols = cols + list(cols_other_lowleft.keys())
    if len(cols_other_lowright) > 0:
        cols = cols + list(cols_other_lowright.keys())
    for col, info in cols_to_label_info.items():
        cols.append(col)
        for lbl_col, vs, styls, lbls in info:
            cols.append(lbl_col)
    cols = list(set(cols))

    data = data.reindex(columns=cols)

    data['t_bkup'] = data[tcol].copy()
    data[tcol] = pd.to_datetime(data[tcol]).map(date2num)
    data['timeI'] = np.arange(0, data.shape[0])

    date_tickers = data['t_bkup'].values

    # 坐标准备
    plt.figure(figsize=figsize)
    if plot_below or len(cols_other_lowleft) > 0:
        gs = GridSpec(3, 1)
        ax1 = plt.subplot(gs[:2, :])
        ax2 = plt.subplot(gs[2, :])
        if len(cols_other_lowright) > 0:
            ax2_ = ax2.twinx()
    else:
        gs = GridSpec(1, 1)
        ax1 = plt.subplot(gs[:, :])
    if len(cols_other_upright) > 0:
        ax1_ = ax1.twinx()

    # 绘制K线图
    data_K = data[['timeI', 'open', 'close', 'high', 'low']].values
    candlestick_ochl(ax=ax1, quotes=data_K, width=width,
                                 colorup='r', colordown='g', alpha=alpha)

    # 标题绘制在K线图顶部
    if title is not None:
        ax1.set_title(title, fontsize=fontsize_label)

    if args_ma and args_boll:
        raise ValueError('均线和布林带只能绘制一种！')

    lns = []

    # 均线图
    if args_ma:
        args_ma = [x for x in args_ma if x < data.shape[0]-1]
        if len(args_ma) > 0:
            for m in args_ma:
                data['MA'+str(m)] = data['close'].rolling(m).mean()
                ln = ax1.plot(data['timeI'], data['MA'+str(m)],
                              label='MA'+str(m))
                lns.append(ln)

    # 布林带
    if args_boll:
        data['boll_mid'] = data['close'].rolling(args_boll[0]).mean()
        data['boll_std'] = data['close'].rolling(args_boll[0]).std()
        data['boll_up'] = data['boll_mid'] + args_boll[1] * data['boll_std']
        data['boll_low'] = data['boll_mid'] - args_boll[1] * data['boll_std']
        ln = ax1.plot(data['timeI'], data['boll_mid'], '-k',
                      label='Boll({a}, {b})'.format(
                            a=args_boll[0], b=args_boll[1]))
        lns.append(ln)
        ax1.plot(data['timeI'], data['boll_low'], '-r')
        ax1.plot(data['timeI'], data['boll_up'], '-g')

    # 上图其它列绘制
    if len(cols_other_upleft) > 0:
        for col, lnsytl_lbl in cols_other_upleft.items():
            lnstyl, lbl = lnsytl_lbl
            if lbl != False:
                if lbl is None:
                    lbl = col
                ln = ax1.plot(data['timeI'], data[col], lnstyl, label=lbl)
                lns.append(ln)
            else:
                ax1.plot(data['timeI'], data[col], lnstyl, label=lbl)
    if len(cols_other_upright) > 0:
        for col, lnsytl_lbl in cols_other_upright.items():
            lnstyl, lbl = lnsytl_lbl
            if lbl != False:
                if lbl is None:
                    lbl = col+'(r)'
                else:
                    lbl = lbl + '(r)'
                ln = ax1_.plot(data['timeI'], data[col], lnstyl, label=lbl)
                lns.append(ln)
            else:
                ax1_.plot(data['timeI'], data[col], lnstyl, label=lbl)

    # 矩形框图
    if rects is not None:
        for (left_low_x, left_low_y), wd, ht in rects:
            ax1.add_patch(patches.Rectangle((left_low_x, left_low_y),
                            wd, ht, fill=False, edgecolor='b', linewidth=2))

    # 上图特殊标注列
    for col, info in cols_to_label_info.items():
        if col in cols_other_upleft or col in ['open', 'high', 'low', 'close']:
            ax_ = ax1
        elif col in cols_other_upright:
            ax_ = ax1_
        else:
            continue
        for lbl_col, vs, styls, lbls in info:
            if lbls != False:
                if lbls is None:
                    lbls = vs
                for k in range(len(vs)):
                    tmp = data[data[lbl_col] == vs[k]][['timeI', col]]
                    ln = ax_.plot(tmp['timeI'], tmp[col], styls[k],
                                  label=lbls[k], markersize=markersize)
                    lns.append(ln)
            else:
                for k in range(len(vs)):
                    tmp = data[data[lbl_col] == vs[k]][['timeI', col]]
                    ax_.plot(tmp['timeI'], tmp[col], styls[k],
                             markersize=markersize)

    # 上图x轴平行线
    for col, parls_info in xparls_info.items():
        if col in cols_other_upleft or col in ['open', 'high', 'low', 'close']:
            ax_ = ax1
        elif col in cols_other_upright:
            ax_ = ax1_
        else:
            continue
        for yval, clor, lnstyl, lnwidth in parls_info:
            ax_.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth)

    # 上图y轴平行线
    if yparls_info_up is not None:
        for xval, clor, lnstyl, lnwidth in yparls_info_up:
            xval = data[data['t_bkup'] == xval]['timeI'].iloc[0]
            ax1.axvline(x=xval, c=clor, ls=lnstyl, lw=lnwidth)

    # 上图坐标轴尺度
    ax1.set_yscale(yscales[0])
    if len(cols_other_upright) > 0:
        ax1_.set_yscale(yscales[1])

    if len(lns) > 0:
        lnsAdd = lns[0]
        for ln in lns[1:]:
            lnsAdd = lnsAdd + ln
        labs = [l.get_label() for l in lnsAdd]
        ax1.legend(lnsAdd, labs, loc=0, fontsize=fontsize_legend)

    # 上图y轴标签文本
    if ylabels[0] is False:
        ax1.set_ylabel(None)
        ax1.set_yticks([])
    else:
        ax1.set_ylabel(ylabels[0], fontsize=fontsize_label)
    if len(cols_other_upright) > 0:
        if ylabels[1] is False:
            ax1_.set_ylabel(None)
            ax1_.set_yticks([])
        else:
            ax1_.set_ylabel(ylabels[1], fontsize=fontsize_label)

    lns = []

    # 底部图
    if plot_below:
        data['up'] = data.apply(lambda x: 1 if x['close'] >= x['open'] else 0,
                                axis=1)
        ax2.bar(data.query('up == 1')['timeI'],
                data.query('up == 1')[plot_below], color='r',
                width=width+0.1, alpha=alpha)
        ax2.bar(data.query('up == 0')['timeI'],
                data.query('up == 0')[plot_below], color='g',
                width=width+0.1, alpha=alpha)
        # ax2.set_ylabel(plot_below, fontsize=fontsize_label)
        ax2.grid(grid)

        # 底部均线图
        if args_ma_below:
            args_ma_below = [x for x in args_ma_below if x < data.shape[0]-1]
            if len(args_ma_below) > 0:
                for m in args_ma_below:
                    data['MA'+str(m)] = data[plot_below].rolling(m).mean()
                    ax2.plot(data['timeI'], data['MA'+str(m)],
                             label='MA'+str(m))
                ax2.legend(loc=0, fontsize=fontsize_legend)

    # 下图其它列绘制
    if len(cols_other_lowleft) > 0:
        for col, lnsytl_lbl in cols_other_lowleft.items():
            lnstyl, lbl = lnsytl_lbl
            if lbl != False:
                if lbl is None:
                    lbl = col
                ln = ax2.plot(data['timeI'], data[col], lnstyl, label=lbl)
                lns.append(ln)
            else:
                ax2.plot(data['timeI'], data[col], lnstyl, label=lbl)
    if len(cols_other_lowright) > 0:
        for col, lnsytl_lbl in cols_other_lowright.items():
            lnstyl, lbl = lnsytl_lbl
            if lbl != False:
                if lbl is None:
                    lbl = col+'(r)'
                else:
                    lbl = lbl + '(r)'
                ln = ax2_.plot(data['timeI'], data[col], lnstyl, label=lbl)
                lns.append(ln)
            else:
                ax2_.plot(data['timeI'], data[col], lnstyl, label=lbl)

    # 下图特殊标注列
    for col, info in cols_to_label_info.items():
        if col in cols_other_lowleft:
            ax_ = ax2
        elif col in cols_other_lowright:
            ax_ = ax2_
        else:
            continue
        for lbl_col, vs, styls, lbls in info:
            if lbls != False:
                if lbls is None:
                    lbls = vs
                for k in range(len(vs)):
                    tmp = data[data[lbl_col] == vs[k]][['timeI', col]]
                    ln = ax_.plot(tmp['timeI'], tmp[col], styls[k],
                                  label=lbls[k], markersize=markersize)
                    lns.append(ln)
            else:
                for k in range(len(vs)):
                    tmp = data[data[lbl_col] == vs[k]][['timeI', col]]
                    ax_.plot(tmp['timeI'], tmp[col], styls[k],
                             markersize=markersize)

    # 下图x轴平行线
    for col, parls_info in xparls_info.items():
        if col in cols_other_lowleft:
            ax_ = ax2
        elif col in cols_other_lowright:
            ax_ = ax2_
        else:
            continue
        for yval, clor, lnstyl, lnwidth in parls_info:
            ax_.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth)

    # 下图y轴平行线
    if plot_below or len(cols_other_lowleft) > 0:
        if yparls_info_low is not None:
            for xval, clor, lnstyl, lnwidth in yparls_info_low:
                xval = data[data['t_bkup'] == xval]['timeI'].iloc[0]
                ax2.axvline(x=xval, c=clor, ls=lnstyl, lw=lnwidth)

    # 下图坐标轴尺度
    if plot_below or len(cols_other_lowleft) > 0:
        ax2.set_yscale(yscales[2])
    if len(cols_other_lowright) > 0:
        ax2_.set_yscale(yscales[3])

    if len(lns) > 0:
        lnsAdd = lns[0]
        for ln in lns[1:]:
            lnsAdd = lnsAdd + ln
        labs = [l.get_label() for l in lnsAdd]
        ax2.legend(lnsAdd, labs, loc=0, fontsize=fontsize_legend)

    # 下图y轴标签文本
    if plot_below or len(cols_other_lowleft) > 0:
        if ylabels[2] is False:
            ax2.set_ylabel(None)
            ax2.set_yticks([])
        else:
            ax2.set_ylabel(ylabels[2], fontsize=fontsize_label)
    if len(cols_other_lowright) > 0:
        if ylabels[3] is False:
            ax2_.set_ylabel(None)
            ax2_.set_yticks([])
        else:
            ax2_.set_ylabel(ylabels[3], fontsize=fontsize_label)

    # x轴刻度
    def format_date(x, pos):
        if x < 0 or x > len(date_tickers)-1:
            return ''
        return date_tickers[int(x)]
    n = data.shape[0]
    xpos = [int(x*n/nXticks) for x in range(0, nXticks)] + [n-1]
    # 上图x轴刻度
    ax1.set_xticks(xpos)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax1.grid(grid)

    # 上图x轴标签文本
    if xlabels[0] is False:
        ax1.set_xlabel(None)
        ax1.set_xticks([])
    else:
        ax1.set_xlabel(xlabels[0], fontsize=fontsize_label)

    # 下图x轴刻度
    if plot_below or len(cols_other_lowleft) > 0:
        ax2.set_xticks(xpos)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax2.grid(grid)

        # 下图x轴标签文本
        if xlabels[1] is False:
            ax2.set_xlabel(None)
            ax2.set_xticks([])
        else:
            ax2.set_xlabel(xlabels[1], fontsize=fontsize_label)

    plt.tight_layout()

    if fig_save_path:
        plt.savefig(fig_save_path)

    plt.show()


if __name__ == '__main__':
    import time
    from dramkit import load_csv
    strt_tm = time.time()

    daily_50etf_pre_fq_path = '../test/510050_daily_pre_fq.csv'
    data = load_csv(daily_50etf_pre_fq_path)
    data['time'] = data['date'].copy()
    data.set_index('time', drop=False, inplace=True)
    tcol = 'time'
    data = data.reindex(columns=['time', 'open', 'low', 'high', 'close',
                                 'volume', 'amount'])
    data['Pavg'] = data['amount'] / data['volume']
    data['Pavg1'] = data[['close', 'high', 'low']].mean(axis=1)
    data['vol2'] = data['volume'].diff()
    data['sig_close'] = 0
    for k in range(5, data.shape[0]-5, 20):
        data.loc[data.index[k], 'sig_close'] = 1
    for k in range(10, data.shape[0]-5, 10):
        data.loc[data.index[k], 'sig_close'] = -1
    data['sig_high'] = 0
    for k in range(8, data.shape[0]-5, 8):
        data.loc[data.index[k], 'sig_high'] = 1
    data['sig_low'] = 0
    for k in range(6, data.shape[0]-5, 10):
        data.loc[data.index[k], 'sig_low'] = -1
    data['sig_avg'] = 0
    for k in range(9, data.shape[0]-5, 11):
        data.loc[data.index[k], 'sig_avg'] = 1
    for k in range(13, data.shape[0]-5, 17):
        data.loc[data.index[k], 'sig_avg'] = -1

    data = data.iloc[-500:, :].copy()

    args_ma = None
    # args_ma = [5, 10, 20, 30, 50]
    # args_boll = None
    args_boll = [15, 2]
    # cols_other_upleft = {}
    cols_other_upleft = {'Pavg1': ('m-', None)}
    cols_other_upright = {'Pavg': ('m-', None)}
    cols_other_lowleft = {'vol2': ('r-', None)}
    # cols_other_lowleft = {}
    cols_other_lowright = {'Pavg1': ('b-', None)}
    # cols_other_lowright = {}
    # cols_to_label_info = {}
    cols_to_label_info = {
            'close': [['sig_close', (-1, 1), ('m^', 'gv'), ('Buy', 'Sel')]],
            'high': [['sig_high', (1,), ('bv',), ('High',)]],
            'low': [['sig_low', (-1,), ('y^',), ('Low',)]],
            # 'vol2': [['sig_avg', (-1, 1), ('yo', 'co'), ('lowA', 'highA')]],
            # 'Pavg1': [['sig_close', (-1, 1), ('m^', 'gv'), ('Buy', 'Sel')]]
            }
    xparls_info = {'Pavg': [(3.7, 'b', '-', 1.5)],
                   'vol2': [(0, 'b', '-', 1.0)],
                   'Pavg1': [(3.9, 'm', '--', 1.5)]}
    yparls_info_up = [('2021-02-05', 'b', '-', 1.5),
                      ('2021-04-14', 'b', '-', 1.5)]
    yparls_info_low = [('2020-12-25', 'b', '-', 1.5),
                       ('2021-03-08', 'b', '-', 1.5)]
    # rects = None
    rects = [[(0, 3.45), 8, 0.1],
             [(19, 3.8), 9, 0.1],
             [(80, 3.45), 10, 0.1]
             ]
    plot_below = None
    # plot_below = 'volume'
    args_ma_below = None
    # args_ma_below = [3, 5, 10]
    ylabels = ['点位', '均价', 'volume', 'voldif']
    xlabels = ['日期', '时间']
    yscales = None
    figsize = (11, 9)
    # fig_save_path = None
    fig_save_path = './test/Candle_test.png'
    title = '50ETF'
    fontsize_label = 20
    fontsize_legend = 15
    width = 0.5
    alpha = 0.95
    grid = True
    markersize = 12
    nXticks = 8

    plot_candle(data, args_ma=args_ma, args_boll=args_boll,
                cols_other_upleft=cols_other_upleft,
                cols_other_upright=cols_other_upright,
                cols_to_label_info=cols_to_label_info, rects=rects,
                xparls_info=xparls_info,
                yparls_info_low=yparls_info_low, yparls_info_up=yparls_info_up,
                plot_below=plot_below, args_ma_below=args_ma_below,
                cols_other_lowleft=cols_other_lowleft,
                cols_other_lowright=cols_other_lowright,
                ylabels=ylabels, xlabels=xlabels, yscales=yscales,
                figsize=figsize, fig_save_path=fig_save_path, title=title,
                fontsize_label=fontsize_label, fontsize_legend=fontsize_legend,
                width=width, alpha=alpha,
                grid=grid, markersize=markersize, nXticks=nXticks)


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
