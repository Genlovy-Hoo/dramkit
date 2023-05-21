# -*- coding: utf-8 -*-

# import seaborn as sns
# sns.set()

import numpy as np

from dramkit.gentools import isnull
from dramkit.gentools import get_con_start_end
from dramkit.gentools import get_update_kwargs
from dramkit.logtools.utils_logger import logger_show

import matplotlib as mpl
mpl.rcParams['font.family'] = ['sans-serif', 'stixgeneral', 'serif']
mpl.rcParams['font.sans-serif'] = ['SimHei', 'KaiTi', 'FangSong']
mpl.rcParams['font.serif'] = ['cmr10', 'SimHei', 'KaiTi', 'FangSong']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm' # 'dejavusans', 'cm', 'stix'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#%%
def _plot_series_with_styls_info(ax, series, styls_info,
                                 lnstyl_default='.-',
                                 lbl_str_ext='',
                                 **kwargs_plot):
    '''
    给定线型设置信息styls_info, 在ax上对series (`pandas.Series`)绘图，
    lnstyl_default设置默认线型    
    styls_info格式形如：('.-b', 'lbl')或'.-b'
    第一种格式中lbl设置图例（legend），lbl若为None则默认取series列名，若为False，则不设置图例
    第二种格式只设置线型，legend默认取series列名
    lbl_str_ext设置legend文本后缀（比如双坐标轴情况下在右轴的legend加上'(右)'）
    **kwargs_plot可接收符合ax.plot函数的其它参数
    '''

    if styls_info is None:
        lnstyl, lbl_str = lnstyl_default, series.name
    else:
        if isinstance(styls_info, str):
            lnstyl, lbl_str = styls_info, series.name
        else:
            if len(styls_info) == 2:
                lnstyl, lbl_str = styls_info
            elif len(styls_info) == 3:
                lnstyl, lbl_str, kwothers = styls_info
                kwargs_plot.update(kwothers)
    lnstyl = lnstyl_default if isnull(lnstyl) else lnstyl
    lbl_str = series.name if lbl_str is None else lbl_str

    if lbl_str is False:
        ax.plot(series, lnstyl, **kwargs_plot)
        return None
    else:
        ln = ax.plot(series, lnstyl, label=str(lbl_str)+lbl_str_ext,
                     **kwargs_plot)
        return ln

#%%
def plot_series(data, cols_styl_up_left, cols_styl_up_right={},
                cols_styl_low_left={}, cols_styl_low_right={},
                cols_to_label_info={}, cols_to_fill_info={},
                col_text_up={}, col_text_low={}, yscales=None,
                xparls_info={}, yparls_info_up=None, yparls_info_low=None,
                fills_yparl_up=None, fills_yparl_low=None, fills_xparl={},
                twinx_align_up=None, twinx_align_low=None,
                ylabels=None, xlabels=None, grids=False, figsize=(11, 7),
                title=None, n_xticks=8, xticks_rotation=None,
                fontsize_label=15, fontsize_title=15, fontsize_text=10,
                fontsize_legend=15, fontsize_tick=10, fontname=None,
                markersize=10, legend_locs=None, fig_save_path=None,
                show=True, logger=None):
    '''
    对data (`pd.DataFrame`)进行多列绘图

    .. note::
        目前功能未考虑data.index重复情况，若有重复可能会导致部分绘图错误

    Parameters
    ----------
    data : pandas.DataFrame
        待作图数据
    cols_styl_up_left : dict
        指定顶部左轴需要绘制的序列及其线型和图例，格式形如：

        ``{'col1': ('.-b', 'lbl1', kwargs), 'col2': ...}`` 或 ``{'col1': '.-b', 'col2': ...}``
        
        第一种格式中 `lbl` 设置图例(legend)，若为None则默认取列名，为False则不设置图例

        第二种格式只设置线型，legend默认取列名
    cols_styl_up_right : dict
        指定顶部右轴需要绘制的序列及其线型和图例，格式同 ``cols_styl_up_left``
    cols_styl_low_left : dict
        指定底部左轴需要绘制的序列及其线型和图例，格式同 ``cols_styl_up_left``
    cols_styl_low_right : dict
        指定底部右轴需要绘制的序列及其线型和图例，格式同 ``cols_styl_up_left``
    cols_to_label_info : dict
        设置需要特殊标注的列绘图信息，格式形如:

        .. code-block:: python

            {col1:
                 [[col_lbl1, (v1, v2, ..), (styl1, styl2, ..), (lbl1, lbl2, ..),
                   {kwargs, v1: {kwargs1}, v2: {kwargs2}, ...}],
                  [col_lbl2, (v1, v2, ..), ...]
                 ],
             col2: ...
            }

        其中col是需要被特殊标注的列，col_lbl为标签列；v指定哪些标签值对应的
        数据用于绘图；styl设置线型；lbl设置图例标签，若为None，则设置为v，若为False，
        则不设置图例标签；{kwargs, v1: {kwargs1}, v2: {kwargs2}}设置其他绘图标注参数
    cols_to_fill_info : dict
        需要进行颜色填充的列信息，格式形如(具体参数key参见matplotlib的fill_between函数):

        ``{col1: {'color': 'c', 'alpha': 0.3}, ...}``
    col_text_up : dict
        上图文本标注设置，格式形如（具体参数key参见matplotlib的ax.text函数）:
            
        ``{col1: (col2, {...}), ...}``
    col_text_low : dict
        下图文本标注设置，格式同 ``col_text_up``
    yscales : None, list
        y轴标轴尺度设置，若为None，则默认普通线性坐标，
        可设置为list指定每个坐标尺度(参见matplotlib中的set_yscale)
    xparls_info : dict
        设置x轴平行线信息，格式形如：

        ``{col1: [(yval1, clor1, styl1, width1, kwargs), (yval2, ...)], col2:, ...}``

        其中yval指定平行线y轴位置，clor设置颜色，styl设置线型，width设置线宽
    yparls_info_up : None, list
        设置顶部x轴平行线格式信息，格式形如：

        ``[(xval1, clor1, styl1, width1, kwargs), (xval2, clor2, style2, width2), ...]``

        其中xval指定平行线x轴位置，clor设置颜色，styl设置线型，width设置线宽
    yparls_info_low : None, list
        设置顶部x轴平行线格式信息，格式同 ``yparls_info_up``
    fills_yparl_up : None, list
        设置上图平行于y轴的填充区域信息，格式形如:

        ``[([x1, x2], clor1, alpha1, kwargs), (...)]``
    fills_yparl_low : None, list
        设置下图平行于y轴的填充区域信息，格式同 ``fills_yparl_up``
    fills_xparl : dict
        设置平行于x轴的填充区域信息，格式形如：

        ``{'col1': [([y1, y2], clor1, alpha1, kwargs), ...], 'col2': ...}``
    twinx_align_up : None, list
        设置上图双坐标轴两边坐标轴刻度对齐位置，格式如 ``[v_left, v_right]`` ，
        绘图时左轴的 ``v_left`` 位置与右轴的 ``v_right`` 位置对齐
    twinx_align_low : None, list
        设置上图双坐标轴两边坐标轴刻度对齐位置，格式同 ``twinx_align_up``
    ylabels : None, list
        设置四个y轴标签文本内容，若为None则不设置标签文本，
        若为False则既不设置y轴标签文本内容，也不显示y轴刻度
    xlabels : None, list
        置两个x轴标签文本内容，若为None则不设置标签文本，
        若为False则既不设置x轴标签文本内容，也不显示x轴刻度
    grids : boll, list
        设置四个坐标轴网格，若grids=True，则在顶部左轴和底部左轴绘制网格；
        若grids=False，则全部没有网格；若为列表，则分别对四个坐标轴设置网格

        .. caution::
            当某个坐标轴设置为不显示刻度时，其对应的网格线也会不显示？
    legend_locs : None, list
        设置上下两个图的legend位置，默认设置为[0, 0]
    fontname : None, str
        字体默认设置为None，可替换其他字体
        (如 ``Courier New``, ``Times New Roman``)

        .. hint::
            matplotlib默认字体为 ``sans-serif``
            
    TODO
    ----
    - 标签指定值时可以设置为函数（考虑是否有必要）
    - 多重索引处理
    - legend位置增加放在图片外面的设置
    - 不规则区域填充设置
    - 添加堆叠图（面积图）绘制方式
    - 数字文本标注增加自定义设置（可针对文本列统一设置，也可针对单个文本设置）
    - 正常绘制与特殊标注重复绘制问题
    - x轴平行线对应列不一定非要在主图绘制列中选择
    - 平行线图层绘制在主线下面
    - 标注图层绘制在线型图层上面（根据输入顺序绘制图层而不是根据坐标轴区域顺序绘制）
    - 上图和下图的x轴不一定非要都是data的index，设置上下图不同x轴坐标
    '''
    
    df = data.copy()

    # 网格设置，grids分别设置顶部左边、顶部右边、底部左边、底部右边的网格
    if grids is True:
        grids = [True, False, True, False]
    elif grids is False or grids is None:
        grids = [False, False, False, False]

    # y轴标签设置
    if ylabels is None:
        ylabels = [None, None, None, None]

    # 坐标轴尺度设置
    if yscales is None:
        yscales = ['linear'] * 4

    # x轴标签设置
    if xlabels is None:
        xlabels = [None, None]

    # legend位置设置
    if legend_locs is None:
        legend_locs = [0, 0]

    # 索引列处理
    if df.index.name is None:
        df.index.name = 'idx'
    idx_name = df.index.name
    if idx_name in df.columns:
        df.drop(idx_name, axis=1, inplace=True)
    df.reset_index(inplace=True)

    if len(cols_styl_low_left) == 0 and len(cols_styl_low_right) > 0:
        logger_show('当底部图只指定右边坐标轴时，默认绘制在左边坐标轴！', logger, 'warning')
        cols_styl_low_left, cols_styl_low_right = cols_styl_low_right, {}

    # 坐标准备
    plt.figure(figsize=figsize)
    if len(cols_styl_low_left) > 0:
        gs = GridSpec(3, 1)
        axUpLeft = plt.subplot(gs[:2, :]) # 顶部为主图，占三分之二高度
        axLowLeft = plt.subplot(gs[2, :])
    else:
        gs = GridSpec(1, 1)
        axUpLeft = plt.subplot(gs[:, :])


    def get_cols_to_label_info(cols_to_label_info, col):
        '''需要进行特殊点标注的列绘图设置信息获取'''

        to_plots = []
        for label_infos in cols_to_label_info[col]:

            if len(label_infos) == 5:
                ext_styl = True
                kwstyl_universal = {}
                kwstyl_unique = {}
                kwstyl = label_infos[4]
                for k, v in kwstyl.items():
                    if not isinstance(v, dict):
                        kwstyl_universal.update({k: v})
                    else:
                        if k in kwstyl_universal.keys():
                            kwstyl_unique[k].update(v)
                        else:
                            kwstyl_unique[k] = v
            else:
                ext_styl = False

            lbl_col = label_infos[0]

            if label_infos[2] is None:
                label_infos = [lbl_col, label_infos[1], [None]*len(label_infos[1]),
                               label_infos[3]]

            if label_infos[3] is False:
                label_infos = [lbl_col, label_infos[1], label_infos[2],
                               [False]*len(label_infos[1])]
            elif isnull(label_infos[3]) or \
                                        all([isnull(x) for x in label_infos[3]]):
                label_infos = [lbl_col, label_infos[1], label_infos[2],
                               label_infos[1]]

            vals = label_infos[1]
            for k in range(len(vals)):
                series = df[df[lbl_col] == vals[k]][col]
                if len(series) > 0:
                    ln_styl = label_infos[2][k]
                    lbl_str = label_infos[3][k]
                    if not ext_styl:
                        to_plots.append([series, (ln_styl, lbl_str)])
                    else:
                        kwothers = {}
                        kwothers.update(kwstyl_universal)
                        if vals[k] in kwstyl_unique.keys():
                            kwothers.update(kwstyl_unique[vals[k]])
                        to_plots.append([series, (ln_styl, lbl_str, kwothers)])

        return to_plots

    def get_parls_info(parlInfo):
        if len(parlInfo) == 5 and isinstance(parlInfo[-1], dict):
            val, clor, lnstyl, lnwidth, kwstyl = parlInfo
        elif len(parlInfo) == 4 and not isinstance(parlInfo[-1], dict):
            val, clor, lnstyl, lnwidth = parlInfo
            kwstyl = {}
        elif len(parlInfo) == 4 and isinstance(parlInfo[-1], dict):
            val, clor, lnstyl, kwstyl = parlInfo
            lnwidth, kwstyl = get_update_kwargs('lw', None, kwstyl, func_update=False)
        elif len(parlInfo) == 3 and not isinstance(parlInfo[-1], dict):
            val, clor, lnstyl = parlInfo
            lnwidth, kwstyl = None, {}
        elif len(parlInfo) == 3 and isinstance(parlInfo[-1], dict):
            val, clor, kwstyl = parlInfo
            lnstyl, kwstyl = get_update_kwargs('ls', None, kwstyl, func_update=False)
            lnwidth, kwstyl = get_update_kwargs('lw', None, kwstyl, func_update=False)
        elif len(parlInfo) == 2 and not isinstance(parlInfo[-1], dict):
            val, clor = parlInfo
            lnstyl, lnwidth, kwstyl = None, None, {}
        elif len(parlInfo) == 2 and isinstance(parlInfo[-1], dict):
            val, kwstyl = parlInfo
            clor, kwstyl = get_update_kwargs('c', None, kwstyl, func_update=False)
            lnstyl, kwstyl = get_update_kwargs('ls', None, kwstyl, func_update=False)
            lnwidth, kwstyl = get_update_kwargs('lw', None, kwstyl, func_update=False)
        else:
            val = parlInfo[0]
            clor, lnstyl, lnwidth, kwstyl = None, None, None, {}
        return val, clor, lnstyl, lnwidth, kwstyl

    def get_xparls_info(parls_info, col, clor_default='k',
                        lnstyl_default='--', lnwidth_default=1.0):
        '''x轴平行线绘图设置信息获取'''
        parls = parls_info[col]
        to_plots = []
        for parlInfo in parls:
            val, clor, lnstyl, lnwidth, kwstyl = get_parls_info(parlInfo)
            clor = clor_default if isnull(clor) else clor
            lnstyl = lnstyl_default if isnull(lnstyl) else lnstyl
            lnwidth = lnwidth_default if isnull(lnwidth) else lnwidth
            to_plots.append([val, clor, lnstyl, lnwidth, kwstyl])
        return to_plots

    def get_yparls_info(parls_info, clor_default='r', lnstyl_default='--',
                        lnwidth_default=1.0):
        '''y轴平行线绘图设置信息获取'''
        to_plots = []
        for parlInfo in parls_info:
            val, clor, lnstyl, lnwidth, kwstyl = get_parls_info(parlInfo)
            clor = clor_default if isnull(clor) else clor
            lnstyl = lnstyl_default if isnull(lnstyl) else lnstyl
            lnwidth = lnwidth_default if isnull(lnwidth) else lnwidth
            val = df[df[idx_name] == val].index[0]
            to_plots.append([val, clor, lnstyl, lnwidth, kwstyl])
        return to_plots

    def get_fill_info(fillInfo):
        if len(fillInfo) == 4 and isinstance(fillInfo[-1], dict):
            locs, clor, alpha, kwstyl = fillInfo
        elif len(fillInfo) == 3 and not isinstance(fillInfo[-1], dict):
            locs, clor, alpha = fillInfo
            kwstyl = {}
        elif len(fillInfo) == 3 and isinstance(fillInfo[-1], dict):
            locs, clor, kwstyl = fillInfo
            alpha, kwstyl = get_update_kwargs('alpha', None, kwstyl, func_update=False)
        elif len(fillInfo) == 2 and not isinstance(fillInfo[-1], dict):
            locs, clor = fillInfo
            alpha, kwstyl = None, {}
        elif len(fillInfo) == 2 and isinstance(fillInfo[-1], dict):
            locs, kwstyl = fillInfo
            clor, kwstyl = get_update_kwargs('color', None, kwstyl, func_update=False)
            alpha, kwstyl = get_update_kwargs('alpha', None, kwstyl, func_update=False)
        else:
            locs = fillInfo[0]
            clor, alpha, kwstyl = None, None, {}
        return locs, clor, alpha, kwstyl

    def get_fills_xparl_info(fills_info, col,
                             clor_default='grey', alpha_default=0.3):
        '''x轴平行填充区域设置信息获取'''
        fills_info_ = fills_info[col]
        to_fills = []
        for fillInfo in fills_info_:
            ylocs, clor, alpha, kwstyl = get_fill_info(fillInfo)
            clor = clor_default if isnull(clor) else clor
            alpha = alpha_default if isnull(alpha) else alpha
            to_fills.append([ylocs, clor, alpha, kwstyl])
        return to_fills

    def get_fills_yparl_info(fills_info, clor_default='grey', alpha_default=0.3):
        '''y轴平行填充区域设置信息获取'''
        to_fills = []
        for fillInfo in fills_info:
            xlocs, clor, alpha, kwstyl = get_fill_info(fillInfo)
            clor = clor_default if isnull(clor) else clor
            alpha = alpha_default if isnull(alpha) else alpha
            xlocs = [df[df[idx_name] == x].index[0] for x in xlocs]
            to_fills.append([xlocs, clor, alpha, kwstyl])
        return to_fills
    
    def get_text_info(text_info, col):
        ''''''
        raise NotImplementedError
    
    def twinx_align(ax_left, ax_right, v_left, v_right):
        '''双坐标轴左右按照v_left和v_right对齐'''
        left_min, left_max = ax_left.get_ybound()
        right_min, right_max = ax_right.get_ybound()
        k = (left_max-left_min) / (right_max-right_min)
        b = left_min - k * right_min
        x_right_new = k * v_right + b
        dif = x_right_new - v_left
        if dif >= 0:
            right_min_new = ((left_min-dif) - b) / k
            k_new = (left_min-v_left) / (right_min_new-v_right)
            b_new = v_left - k_new * v_right
            right_max_new = (left_max - b_new) / k_new
        else:
            right_max_new = ((left_max-dif) - b) / k
            k_new = (left_max-v_left) / (right_max_new-v_right)
            b_new = v_left - k_new * v_right
            right_min_new = (left_min - b_new) / k_new
        def _forward(x):
            return k_new * x + b_new
        def _inverse(x):
            return (x - b_new) / k_new
        ax_right.set_ylim([right_min_new, right_max_new])
        ax_right.set_yscale('function', functions=(_forward, _inverse))
        return ax_left, ax_right


    # lns存放双坐标legend信息
    # 双坐标轴legend参考：https://www.cnblogs.com/Atanisi/p/8530693.html
    lns = []
    # 顶部左边坐标轴
    for col, styl in cols_styl_up_left.items():
        ln = _plot_series_with_styls_info(axUpLeft, df[col], styl)
        if ln is not None:
            lns.append(ln)

        # 填充
        if col in cols_to_fill_info.keys():
            kwargs_fill = cols_to_fill_info[col]
            axUpLeft.fill_between(df.index, df[col], **kwargs_fill)

        # 特殊点标注
        if col in cols_to_label_info.keys():
            to_plots = get_cols_to_label_info(cols_to_label_info, col)
            for series, styls_info in to_plots:
                ln = _plot_series_with_styls_info(axUpLeft, series, styls_info,
                                    lnstyl_default='ko', markersize=markersize)
                if ln is not None:
                    lns.append(ln)

        # x轴平行线
        if col in xparls_info.keys():
            to_plots = get_xparls_info(xparls_info, col)
            for yval, clor, lnstyl, lnwidth, kwstyl_ in to_plots:
                axUpLeft.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth,
                                 **kwstyl_)

        # x轴平行填充
        xlimMinUp, xlimMaxUp = axUpLeft.axis()[0], axUpLeft.axis()[1]
        if col in fills_xparl.keys():
            to_fills = get_fills_xparl_info(fills_xparl, col)
            for ylocs, clor, alpha, kwstyl_ in to_fills:
                axUpLeft.fill_betweenx(ylocs, xlimMinUp, xlimMaxUp,
                                  color=clor, alpha=alpha, **kwstyl_)
        
        # 文本标注
        if col in col_text_up:
            col_text = col_text_up[col][0]
            for idx, val in df[col_text].to_dict().items():
                if not isnull(val):
                    yloc = df.loc[idx, col]
                    axUpLeft.text(idx, yloc, val,
                                  ha='center', va='bottom',
                                  fontsize=fontsize_text)

    # 坐标轴尺度
    axUpLeft.set_yscale(yscales[0])

    # y轴平行线
    if not isnull(yparls_info_up):
        to_plots = get_yparls_info(yparls_info_up)
        for xval, clor, lnstyl, lnwidth, kwstyl_ in to_plots:
            axUpLeft.axvline(x=xval, c=clor, ls=lnstyl, lw=lnwidth,
                             **kwstyl_)

    # y轴平行填充
    if not isnull(fills_yparl_up):
        ylimmin, ylimmax = axUpLeft.axis()[2], axUpLeft.axis()[3]
        to_fills = get_fills_yparl_info(fills_yparl_up)
        for xlocs, clor, alpha, kwstyl_ in to_fills:
            axUpLeft.fill_between(xlocs, ylimmin, ylimmax,
                                  color=clor, alpha=alpha, **kwstyl_)

    # 顶部左边坐标轴网格
    axUpLeft.grid(grids[0])

    # 标题绘制在顶部图上
    if title is not None:
        if isnull(fontname):
            axUpLeft.set_title(title, fontsize=fontsize_title)
        else:
            axUpLeft.set_title(title, fontdict={'family': fontname,
                                                'size': fontsize_title})

    # y轴标签文本
    if ylabels[0] is False:
        axUpLeft.set_ylabel(None)
        axUpLeft.set_yticks([])
    else:
        if isnull(fontname):
            axUpLeft.set_ylabel(ylabels[0], fontsize=fontsize_label)
            [_.set_fontsize(fontsize_tick) for _ in axUpLeft.get_yticklabels()]
        else:
            axUpLeft.set_ylabel(ylabels[0], fontdict={'family': fontname,
                                                      'size': fontsize_label})
            # y轴刻度字体
            [_.set_fontname(fontname) for _ in axUpLeft.get_yticklabels()]
            [_.set_fontsize(fontsize_tick) for _ in axUpLeft.get_yticklabels()]

    # 顶部右边坐标轴
    if len(cols_styl_up_right) > 0:
        axUpRight = axUpLeft.twinx()
        for col, styl in cols_styl_up_right.items():
            ln = _plot_series_with_styls_info(axUpRight, df[col], styl,
                                             lbl_str_ext='(右)')
            if ln is not None:
                lns.append(ln)

            # 填充
            if col in cols_to_fill_info.keys():
                kwargs_fill = cols_to_fill_info[col]
                axUpRight.fill_between(df.index, df[col], **kwargs_fill)

            # 特殊点标注
            if col in cols_to_label_info.keys():
                to_plots = get_cols_to_label_info(cols_to_label_info, col)
                for series, styls_info in to_plots:
                    ln = _plot_series_with_styls_info(axUpRight, series,
                                            styls_info, lnstyl_default='ko',
                                    markersize=markersize, lbl_str_ext='(右)')
                    if ln is not None:
                        lns.append(ln)

            # x轴平行线
            if col in xparls_info.keys():
                to_plots = get_xparls_info(xparls_info, col)
                for yval, clor, lnstyl, lnwidth, kwstyl_ in to_plots:
                    axUpRight.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth,
                                      **kwstyl_)

            # x轴平行填充
            if col in fills_xparl.keys():
                to_fills = get_fills_xparl_info(fills_xparl, col)
                for ylocs, clor, alpha, kwstyl_ in to_fills:
                    axUpRight.fill_betweenx(ylocs, xlimMinUp, xlimMaxUp,
                                      color=clor, alpha=alpha, **kwstyl_)

        # 顶部双坐标轴刻度对齐
        if twinx_align_up is not None:
            axUpLeft, axUpRight = twinx_align(axUpLeft, axUpRight,
                                    twinx_align_up[0], twinx_align_up[1])

        # 坐标轴尺度
        axUpRight.set_yscale(yscales[1])

        # 顶部右边坐标轴网格
        axUpRight.grid(grids[1])

        # y轴标签文本
        if ylabels[1] is False:
            axUpRight.set_ylabel(None)
            axUpRight.set_yticks([])
        else:
            if isnull(fontname):
                axUpRight.set_ylabel(ylabels[1], fontsize=fontsize_label)
                [_.set_fontsize(fontsize_tick) for _ in axUpRight.get_yticklabels()]
            else:
                axUpRight.set_ylabel(ylabels[1], fontdict={'family': fontname,
                                                           'size': fontsize_label})
                # y轴刻度字体
                [_.set_fontname(fontname) for _ in axUpRight.get_yticklabels()]
                [_.set_fontsize(fontsize_tick) for _ in axUpRight.get_yticklabels()]

    # 顶部图legend合并显示
    if len(lns) > 0:
        lnsAdd = lns[0]
        for ln in lns[1:]:
            lnsAdd = lnsAdd + ln
        labs = [l.get_label() for l in lnsAdd]
        if isnull(fontname):
            axUpLeft.legend(lnsAdd, labs, loc=legend_locs[0],
                            fontsize=fontsize_legend)
        else:
            axUpLeft.legend(lnsAdd, labs, loc=legend_locs[0],
                            prop={'family': fontname, 'size': fontsize_legend})


    if len(cols_styl_low_left) > 0:
        # 要绘制底部图时取消顶部图x轴刻度
        # axUpLeft.set_xticks([]) # 这样会导致设置网格线时没有竖线
        axUpLeft.set_xticklabels([]) # 这样不会影响设置网格
        lns = []

        # 底部左边坐标轴
        for col, styl in cols_styl_low_left.items():
            ln = _plot_series_with_styls_info(axLowLeft, df[col], styl)
            if ln is not None:
                lns.append(ln)

            # 填充
            if col in cols_to_fill_info.keys():
                kwargs_fill = cols_to_fill_info[col]
                axLowLeft.fill_between(df.index, df[col], **kwargs_fill)

            # 特殊点标注
            if col in cols_to_label_info.keys():
                to_plots = get_cols_to_label_info(cols_to_label_info, col)
                for series, styls_info in to_plots:
                    ln = _plot_series_with_styls_info(axLowLeft, series,
                        styls_info, lnstyl_default='ko', markersize=markersize)
                    if ln is not None:
                        lns.append(ln)

            # x轴平行线
            if col in xparls_info.keys():
                to_plots = get_xparls_info(xparls_info, col)
                for yval, clor, lnstyl, lnwidth, kwstyl_ in to_plots:
                    axLowLeft.axhline(y=yval, c=clor, ls=lnstyl, lw=lnwidth,
                                      **kwstyl_)

            # x轴平行填充
            xlimMinLow, xlimMaxLow = axLowLeft.axis()[0], axLowLeft.axis()[1]
            if col in fills_xparl.keys():
                to_fills = get_fills_xparl_info(fills_xparl, col)
                for ylocs, clor, alpha, kwstyl_ in to_fills:
                    axLowLeft.fill_betweenx(ylocs, xlimMinLow, xlimMaxLow,
                                      color=clor, alpha=alpha, **kwstyl_)

        # 坐标轴尺度
        axLowLeft.set_yscale(yscales[2])

        # y轴平行线
        if not isnull(yparls_info_low):
            to_plots = get_yparls_info(yparls_info_low)
            for xval, clor, lnstyl, lnwidth, kwstyl_ in to_plots:
                axLowLeft.axvline(x=xval, c=clor, ls=lnstyl, lw=lnwidth,
                                  **kwstyl_)

        # y轴平行填充
        if not isnull(fills_yparl_low):
            ylimmin, ylimmax = axLowLeft.axis()[2], axLowLeft.axis()[3]
            to_fills = get_fills_yparl_info(fills_yparl_low)
            for xlocs, clor, alpha, kwstyl_ in to_fills:
                axLowLeft.fill_between(xlocs, ylimmin, ylimmax,
                                       color=clor, alpha=alpha, **kwstyl_)

        # 底部左边坐标轴网格
        axLowLeft.grid(grids[2])

        # y轴标签文本
        if ylabels[2] is False:
            axLowLeft.set_ylabel(None)
            axLowLeft.set_yticks([])
        else:
            if isnull(fontname):
                axLowLeft.set_ylabel(ylabels[2], fontsize=fontsize_label)
                [_.set_fontsize(fontsize_tick) for _ in axLowLeft.get_yticklabels()]
            else:
                axLowLeft.set_ylabel(ylabels[2], fontdict={'family': fontname,
                                                           'size': fontsize_label})
                # y轴刻度字体
                [_.set_fontname(fontname) for _ in axLowLeft.get_yticklabels()]
                [_.set_fontsize(fontsize_tick) for _ in axLowLeft.get_yticklabels()]

        # 底部右边坐标轴
        if len(cols_styl_low_right) > 0:
            axLowRight = axLowLeft.twinx()
            for col, styl in cols_styl_low_right.items():
                ln = _plot_series_with_styls_info(axLowRight, df[col], styl,
                                                 lbl_str_ext='(右)')
                if ln is not None:
                    lns.append(ln)

                # 填充
                if col in cols_to_fill_info.keys():
                    kwargs_fill = cols_to_fill_info[col]
                    axLowRight.fill_between(df.index, df[col], **kwargs_fill)

                # 特殊点标注
                if col in cols_to_label_info.keys():
                    to_plots = get_cols_to_label_info(cols_to_label_info, col)
                    for series, styls_info in to_plots:
                        ln = _plot_series_with_styls_info(axLowRight, series,
                                            styls_info, lnstyl_default='ko',
                                    markersize=markersize, lbl_str_ext='(右)')
                        if ln is not None:
                            lns.append(ln)

                # x轴平行线
                if col in xparls_info.keys():
                    to_plots = get_xparls_info(xparls_info, col)
                    for yval, clor, lnstyl, lnwidth, kwstyl_ in to_plots:
                        axLowRight.axhline(y=yval, c=clor, ls=lnstyl,
                                           lw=lnwidth, **kwstyl_)

                # x轴平行填充
                if col in fills_xparl.keys():
                    to_fills = get_fills_xparl_info(fills_xparl, col)
                    for ylocs, clor, alpha, kwstyl_ in to_fills:
                        axLowRight.fill_betweenx(ylocs, xlimMinUp, xlimMaxUp,
                                          color=clor, alpha=alpha, **kwstyl_)

            # 底部双坐标轴刻度对齐
            if twinx_align_low is not None:
                axLowLeft, axLowRight = twinx_align(axUpLeft, axUpRight,
                                      twinx_align_low[0], twinx_align_low[1])
            
            # 坐标轴尺度
            axLowRight.set_yscale(yscales[3])

            # 底部右边坐标轴网格
            axLowRight.grid(grids[3])

            # y轴标签文本
            if ylabels[3] is False:
                axLowRight.set_ylabel(None)
                axLowRight.set_yticks([])
            else:
                if isnull(fontname):
                    axLowRight.set_ylabel(ylabels[3], fontsize=fontsize_label)
                    [_.set_fontsize(fontsize_tick) for _ in axLowRight.get_yticklabels()]
                else:
                    axLowRight.set_ylabel(ylabels[3],
                                          fontdict={'family': fontname,
                                                    'size': fontsize_label})
                    # y轴刻度字体
                    [_.set_fontname(fontname) for _ in axLowRight.get_yticklabels()]
                    [_.set_fontsize(fontsize_tick) for _ in axLowRight.get_yticklabels()]

        # 底部图legend合并显示
        if len(lns) > 0:
            lnsAdd = lns[0]
            for ln in lns[1:]:
                lnsAdd = lnsAdd + ln
            labs = [l.get_label() for l in lnsAdd]
            if isnull(fontname):
                axLowLeft.legend(lnsAdd, labs, loc=legend_locs[1],
                                 fontsize=fontsize_legend)
            else:
                axLowLeft.legend(lnsAdd, labs, loc=legend_locs[1],
                                 prop={'family': fontname, 'size': fontsize_legend})

    # x轴刻度
    n = df.shape[0]
    xpos = [int(x*n/n_xticks) for x in range(0, n_xticks)] + [n-1]
    # 上图x轴刻度
    axUpLeft.set_xticks(xpos)
    if isnull(fontname):
        axUpLeft.set_xticklabels([df.loc[x, idx_name] for x in xpos],
                                  fontsize=fontsize_tick,
                                  rotation=xticks_rotation)
    else:
        axUpLeft.set_xticklabels([df.loc[x, idx_name] for x in xpos],
                                 fontdict={'family': fontname,
                                           'size': fontsize_tick},
                                 rotation=xticks_rotation)
    # 下图x轴刻度
    if len(cols_styl_low_left) > 0:
        axLowLeft.set_xticks(xpos)
        if isnull(fontname):
            axLowLeft.set_xticklabels([df.loc[x, idx_name] for x in xpos],
                                      fontsize=fontsize_tick,
                                      rotation=xticks_rotation)

        else:
            axLowLeft.set_xticklabels([df.loc[x, idx_name] for x in xpos],
                                      fontdict={'family': fontname,
                                                'size': fontsize_tick},
                                      rotation=xticks_rotation)

    # x轴标签文本-上图
    if xlabels[0] is False:
        axUpLeft.set_xlabel(None)
        axUpLeft.set_xticks([])
    else:
        if isnull(fontname):
            axUpLeft.set_xlabel(xlabels[0], fontsize=fontsize_label)
        else:
            axUpLeft.set_xlabel(xlabels[0], fontdict={'family': fontname,
                                                      'size': fontsize_label})
    # x轴标签文本-下图
    if len(cols_styl_low_left) > 0:
        if xlabels[1] is False:
            axLowLeft.set_xlabel(None)
            axLowLeft.set_xticks([])
        else:
            if isnull(fontname):
                axLowLeft.set_xlabel(xlabels[1], fontsize=fontsize_label)
            else:
                axLowLeft.set_xlabel(xlabels[1], fontdict={'family': fontname,
                                                           'size': fontsize_label})

    plt.tight_layout()

    # 保存图片
    if fig_save_path:
        plt.savefig(fig_save_path)

    if not show:
        plt.close()
    else:
        plt.show()

#%%
def plot_series_conlabel(data, conlabel_info, del_repeat_lbl=True, **kwargs):
    '''
    在 :func:`dramkit.plottools.plot_common.plot_series` 基础上添加了连续标注绘图功能

    Parameters
    ----------
    data : pandas.DataFrame
        待作图数据
    conlabel_info : dict
        需要进行连续标注的列绘图信息，格式形如：

        ``{col: [[lbl_col, (v1, ...), (styl1, ...), (lbl1, ...)]]}``
        
        .. note::
            (v1, ...)中的最后一个值会被当成默认值，其余的当成特殊值
            （绘图时为了保证连续会将默认值与特殊值连接起来）
    del_repeat_lbl : bool
        当conlabel_info与cols_to_label_info存在重复设置信息时，
        是否删除cols_to_label_info中的设置信息
    **kwargs :
        :func:`dramkit.plottools.plot_common.plot_series` 接受的参数
    '''

    df_ = data.copy()
    df_['_tmp_idx_'] = range(0, df_.shape[0])

    kwargs_new = kwargs.copy()
    if 'cols_to_label_info' in kwargs_new.keys():
        cols_to_label_info = kwargs_new['cols_to_label_info']
    else:
        cols_to_label_info = {}

    def _deal_exist_lbl_col(col, lbl_col, del_exist=True):
        '''
        处理cols_to_label_info中已经存在的待标注列，
        del_exist为True时删除重复的
        '''
        if col in cols_to_label_info.keys():
            if len(cols_to_label_info[col]) > 0 and del_exist:
                for k in range(len(cols_to_label_info[col])):
                    if cols_to_label_info[col][k][0] == lbl_col:
                        del cols_to_label_info[col][k]
        else:
            cols_to_label_info[col] = []

    for col, lbl_infos in conlabel_info.items():
        lbl_infos_new = []
        for lbl_info in lbl_infos:
            lbl_col = lbl_info[0]
            _deal_exist_lbl_col(col, lbl_col, del_exist=del_repeat_lbl)
            Nval = len(lbl_info[1])
            tmp = 0
            for k in range(0, Nval):
                val = lbl_info[1][k]
                start_ends = get_con_start_end(df_[lbl_col], lambda x: x == val)
                for _ in range(0, len(start_ends)):
                    new_col = '_'+lbl_col+'_tmp_'+str(tmp)+'_'
                    df_[new_col] = np.nan
                    idx0, idx1 = start_ends[_][0], start_ends[_][1]+1
                    if k == Nval-1:
                        idx0, idx1 = max(0, idx0-1), min(idx1+1, df_.shape[0])
                    df_.loc[df_.index[idx0: idx1], new_col] = val
                    if _ == 0:
                        if len(lbl_info) == 4:
                            lbl_infos_new.append([new_col, (val,),
                                         (lbl_info[2][k],), (lbl_info[3][k],)])
                        elif len(lbl_info) == 5:
                            lbl_infos_new.append([new_col, (val,),
                            (lbl_info[2][k],), (lbl_info[3][k],), lbl_info[4]])
                    else:
                        if len(lbl_info) == 4:
                            lbl_infos_new.append([new_col, (val,),
                                          (lbl_info[2][k],), (False,)])
                        elif len(lbl_info) == 5:
                            lbl_infos_new.append([new_col, (val,),
                                    (lbl_info[2][k],), (False,), lbl_info[4]])
                    tmp += 1
        # cols_to_label_info[col] += lbl_infos_new
        cols_to_label_info[col] = lbl_infos_new + cols_to_label_info[col]

    kwargs_new['cols_to_label_info'] = cols_to_label_info

    plot_series(df_, **kwargs_new)

#%%
def plot_maxmins(data, col, col_label, label_legend=['Max', 'Min'],
                 figsize=(11, 6), grid=True, title=None, n_xticks=8,
                 markersize=10, fig_save_path=None, **kwargs):
    '''
    | 绘制序列数据(data中col指定列)并标注极大极小值点
    | col_label指定列中值1表示极大值点，-1表示极小值点，0表示普通点
    | label_legend指定col_label为1和-1时的图标标注
    | \\**kwargs为 :func:`dramkit.plottools.plot_common.plot_series` 支持的其它参数
    '''
    plot_series(data, {col: ('-k.', None)},
                cols_to_label_info={col: [[col_label, (1, -1), ('bv', 'r^'),
                                          label_legend]]},
                grids=grid, figsize=figsize, title=title, n_xticks=n_xticks,
                markersize=markersize, fig_save_path=fig_save_path, **kwargs)

#%%
def _plot_maxmins_bk(data, col, col_label, label_legend=['Max', 'Min'],
                     figsize=(11, 6), grid=True, title=None, n_xticks=8,
                     markersize=10, fontsize=15, fig_save_path=None):
    '''
    绘制序列数据(data中col指定列)并标注极大极小值点
    col_label指定列中值1表示极大值点，-1表示极小值点，0表示普通点
    label_legend指定col_label为1和-1时的图标标注
    n_xticks设置x轴刻度显示数量
    '''

    df = data.copy()
    if df.index.name is None:
        df.index.name = 'idx'
    idx_name = df.index.name
    if idx_name in df.columns:
        df.drop(idx_name, axis=1, inplace=True)
    df.reset_index(inplace=True)

    series = df[col]
    series_max = df[df[col_label] == 1][col]
    series_min = df[df[col_label] == -1][col]

    plt.figure(figsize=figsize)
    plt.plot(series, '-k.', label=col)
    plt.plot(series_max, 'bv', markersize=markersize, label=label_legend[0])
    plt.plot(series_min, 'r^', markersize=markersize, label=label_legend[1])
    plt.legend(loc=0, fontsize=fontsize)

    n = df.shape[0]
    xpos = [int(x*n/n_xticks) for x in range(0, n_xticks)] + [n-1]
    plt.xticks(xpos, [df.loc[x, idx_name] for x in xpos])

    plt.grid(grid)

    if title:
        plt.title(title, fontsize=fontsize)

    if fig_save_path:
        plt.savefig(fig_save_path)

    plt.show()

#%%
if __name__ == '__main__':
    import pandas as pd
    from dramkit.gentools import TimeRecoder
    
    tr = TimeRecoder()

    #%%
    col1 = np.random.normal(10, 5, (100, 1))
    col2 = np.random.rand(100, 1)
    col3 = np.random.uniform(0, 20, (100, 1))
    col4 = col1 ** 2

    df = pd.DataFrame(np.concatenate((col1, col2, col3, col4), axis=1))
    df.columns = ['col1', 'col2', 'col3', 'col4']
    df['label1'] = df['col1'].apply(lambda x: 1 if x > 15 else \
                                                        (-1 if x < 5 else 0))
    df['label2'] = df['col3'].apply(lambda x: 1 if x > 15 else \
                                                        (-1 if x < 5 else 0))
    df.index = list(map(lambda x: 'idx'+str(x), df.index))


    plot_maxmins(df, 'col1', 'label1', label_legend=['high', 'low'],
                  figsize=(11, 7), grid=False, title='col1', n_xticks=20,
                  markersize=10, fig_save_path=None)


    plot_series(df, {'col1': ('.-r', None)},
                cols_styl_up_right={'col2': ('.-y', 0),
                                    'col3': ('-3', '3')},
                # cols_styl_low_left={'col1': ('.-r', 't1')},
                cols_styl_low_right={'col4': ('.-k', 't4')},
                cols_to_label_info={'col2':
                                [['label1', (1, -1), ('gv', 'r^'), None]],
                                    'col4':
                                [['label2', (-1, 1), ('b*', 'mo'), None]]},
                yscales=None,
                xparls_info={'col1': [(10, 'k', '--', 3), (15, 'b', '-', 1)],
                              'col4': [(200, None, None, None)]},
                yparls_info_up=[('idx20', None, None, None),
                                ('idx90', 'g', '-', 4)],
                yparls_info_low=[('idx50', None, None, None),
                                  ('idx60', 'b', '--', 2)],
                fills_yparl_up=[(['idx2', 'idx12'], 'black', 0.5),
                                (['idx55', 'idx77'], None, None)],
                fills_yparl_low=[(['idx22', 'idx32'], 'red', 0.5),
                                 (['idx65', 'idx87'], None, None, {}),
                                 (['idx37', 'idx50'], None, None, {})],
                fills_xparl={'col1': [([20, 25], 'green', 0.5),
                                      ([0, 5], None, None, {})],
                             'col2': [([10, 12.5], 'blue', 0.5)],
                             'col3': [([5.5, 8.5], 'red', 0.5)],
                             'col4': [([200, 400], 'yellow', None, {}),
                                      ([0, 100], 'green', None, {})]},
                ylabels=['y1', 'y2', None, False],
                xlabels=['$X_1$', '$x^2$'],
                grids=[True, False, True, True], figsize=(10, 8),
                title='test', n_xticks=8,
                # fontname='Times New Roman',
                xticks_rotation=45,
                fontsize_label=15, fontsize_title=15, fontsize_legend=15,
                fontsize_tick=15, markersize=10, logger=None,
                fig_save_path='./_test/plot_common.png')
    plot_series(df, {'col1': ('.-r', None)},
                # cols_to_label_info={'col1': [['label1', (1, -1), ('gv', 'r^'),
                #             None], ['label2', (-1, 1), ('*', 'o'), None]]},
                cols_to_label_info=\
                    {'col1': [
                        ['label1', (1, -1), ('gv', 'r^'), None, {'alpha': 0.5}],
                        ['label2', (-1, 1), ('*', 'o'), None,
                         {'markersize': 20, -1: {'alpha': 1}, 1: {'alpha': 0.3}}]
                    ]},
                yscales=None,
                xparls_info={'col1': [(10, 'k', '--', 5, {'alpha': 0.3}),
                                      (15, None, None, None)],
                             'col4': [(200, None, None, None)]},
                yparls_info_up=[('idx20', None, None, None),
                                ('idx90', 'g', '-', 5, {'alpha': 0.5})],
                yparls_info_low=[('idx50', None, None, None),
                                 ('idx60', 'b', '--', 5)],
                ylabels=['a', '2', None, False],
                grids=False, figsize=(10, 8), title='test', n_xticks=10,
                fontsize_label=30, markersize=10,
                fig_save_path='./_test/plot_common.png', logger=None)

    #%%
    df1 = pd.DataFrame({'col': [1, 10, 100, 10, 100, 10000, 100]})
    plot_series(df1, {'col': '.-k'})
    plot_series(df1, {'col': '.-k'}, yscales=['log'])

    #%%
    df2 = pd.DataFrame({'y1': [1, 2, 3, 1, 5, 6, 7],
                        'y2': [0.0, -0.1, -0.2, -0.25, np.nan, -0.2, -0.05],
                        'y3': [2, 3, 4, 2, 6, 7, 8],})
    plot_series(df2, {'y1': '.-k', 'y3': '.-y'},
                cols_styl_up_right={'y2': ('-b', None, {'alpha': 0.4})},
                cols_to_fill_info={
                    'y2': {'color': 'c', 'alpha': 0.3},
                    # 'y1': {'color': 'c', 'alpha': 0.3},
                    # 'y3': {'color': 'm', 'alpha': 0.5}
                    }
                )

    #%%
    df3 = pd.DataFrame({'x': np.random.normal(10, 5, (100,))})
    df3['label0'] = 0
    df3.loc[df3.index[[2, 20, 30, 90]], 'label0'] = 1
    df3.loc[df3.index[[5, 26, 40, 70]], 'label0'] = -1
    df3['label'] = 0
    df3.loc[df3.index[5:20], 'label'] = 1
    df3.loc[df3.index[30:50], 'label'] = -1
    df3.loc[df3.index[60:80], 'label'] = 1
    df3['x1'] = df3['x'] - 5

    plot_series_conlabel(df3,
                          # conlabel_info={},
                          conlabel_info={'x': [['label', (1, -1), ('.-r', '.-b'), (None, None),
                                  {'alpha': 1, 1: {'markersize': 20}}]]},
                          cols_styl_up_left={'x': '.-k'},
                          cols_to_label_info={'x':
                            [['label', (-1, 1), ('r^', 'gv'), False]]},
                          del_repeat_lbl=False,
                          # cols_to_fill_info={
                          #   'x': {'y2': df3['x'].min(),
                          #         'color': 'c', 'alpha': 0.3}}
                          # cols_to_fill_info={
                          #   'x': {'y2': df3['x'].max(),
                          #         'color': 'c', 'alpha': 0.3}}
                          cols_to_fill_info={
                            'x': {'y2': df3['x1'],
                                  'color': 'c', 'alpha': 0.3}},
                          xticks_rotation=45)

    #%%
    tr.used()
