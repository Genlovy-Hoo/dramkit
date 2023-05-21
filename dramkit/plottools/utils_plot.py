# -*- coding: utf-8 -*-


def twinx_align(ax_left, ax_right, v_left, v_right):
    '''
    实现左轴(ax_left)在v_left位置和右轴(ax_right)在v_right位置对齐
    '''
    left_min, left_max = ax_left.get_ybound()
    right_min, right_max = ax_right.get_ybound()
    k = (left_max-left_min) / (right_max-right_min)
    b = left_min - k * right_min
    x_right_new = k * v_right + b
    dif = x_right_new - v_left
    # 第一次线性映射，目的是获取左轴和右轴的伸缩比例，通过这层映射，
    # 计算可以得到右轴指定位置映射到左轴之后在左轴的位置。
    # 右轴目标映射到左轴的位置与左轴目标位置存在一个差值，
    # 这个差值就是右轴的一端需要扩展的距离，这个距离是映射到左轴之后按左轴的尺度度量的。
    # 通过第一次线性映射的逆映射，计算得到右轴一端实际需要扩展的距离。
    # 得到右轴一端的扩展距离之后，右轴就有两个固定点：一个端点和一个目标点。
    # 将右轴这两个固定点与左轴对应的点做第二次线性映射，可以再次得到两轴的伸缩比例，
    # 得到新的伸缩比例之后，通过左轴的另一个端点进行逆映射，可以计算题右轴的另一个端点。
    # 最后通过右轴两个端点位置以及新的伸缩比例对右轴进行伸缩变化，
    # 即可将左轴与右轴在指定刻度位置对齐。
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


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from dramkit import plot_series

    # 数据
    # 每日净流入
    net_in = [100, 0, 0, 0, 20, 0, 0, 0, 0, -20,
              0, 0, 30, 0, 0, 0, 0, 0, -30, 0]
    # 每日盈亏金额
    net_gain= [0, -2, -3, 5, 2, 3, 4, 5, 5, -1,
               -4, -10, 2, 5, 9, 6, 0, 1, -1, 9]
    df = pd.DataFrame({'net_in': net_in, 'net_gain': net_gain})
    
    df['total_in'] = df['net_in'].cumsum()
    df['value'] = df['total_in'] + df['net_gain'].cumsum()
    
    df['value_net'] = df['value'] / df['value'].iloc[0] # 每日净值  
    df['pct'] = df['value'] / df['total_in'] - 1 # 实际累计盈亏比例
    
    
    # 普通作图
    plot_series(df,
                {'value_net': ('-k', False)},
                cols_styl_up_right={'pct': ('-b', False)},
                xparls_info={'value_net': [(1,)], 'pct': [(0,)]},
                ylabels=['净值', '总盈亏率'], title='普通作图')
    
    # 对齐作图
    plot_series(df,
                {'value_net': ('-k', False)},
                cols_styl_up_right={'pct': ('-b', False)},
                xparls_info={'value_net': [(1,)], 'pct': [(0,)]},
                twinx_align_up=[1, 0],
                ylabels=['净值', '总盈亏率'], title='对齐作图')
    
    
    # from matplotlib import pyplot as plt
    
    # # 普通作图
    # plt.figure(figsize=(10, 7.5))
    # ax1 = plt.subplot(111)
    # ax1.plot(df['value_net'], '-k')
    # ax1.axhline(1, c='k', lw=1, ls='--')
    # ax1.set_ylabel('净值', fontsize=16)
    # ax2 = ax1.twinx()
    # ax2.plot(df['pct'], '-b')
    # ax2.axhline(0, c='k', lw=1, ls='--')
    # ax2.set_ylabel('总盈亏率', fontsize=16)
    # plt.title('普通作图', fontsize=16)
    # plt.show()
    
    # # 双坐标轴刻度对齐作图
    # plt.figure(figsize=(10, 7.5))
    # ax1 = plt.subplot(111)
    # ax1.plot(df['value_net'], '-k')
    # ax1.axhline(1, c='k', lw=1, ls='--')
    # ax1.set_ylabel('净值', fontsize=16)
    # ax2 = ax1.twinx()
    # ax2.plot(df['pct'], '-b')
    # ax2.axhline(0, c='k', lw=1, ls='--')
    # ax2.set_ylabel('总盈亏率', fontsize=16)
    # twinx_align(ax1, ax2, 1, 0)
    # plt.title('净值1和盈亏率0对齐作图', fontsize=16)
    # plt.show()


    # plot_series文本标注测试
    df['text_up'] = np.nan
    df.loc[4, 'text_up'] = 'CashIn'
    df.loc[9, 'text_up'] = '转出'
    plot_series(df,
                {'value_net': ('-k', False)},
                cols_styl_up_right={'pct': ('-b', False)},
                xparls_info={'value_net': [(1,)], 'pct': [(0,)]},
                col_text_up={'value_net': ('text_up',)},
                # col_text_up={'value_net': ('value_net',)},
                fontsize_text=15,
                ylabels=['净值', '总盈亏率'], title='普通作图')





