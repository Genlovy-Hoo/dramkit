# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


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


if __name__ == '__main__':
    
    left = [10, 15, 20]
    right = [5, 20, 30]
    x_left = 15
    x_right = 25
    
    
    # left = [0.9, 1.0, 0.9699418733176421, 1.2,
    #         0.953508406912784, 0.9539960468654505]
    # right = [-1, 0.0, -0.010064840127400602, 3.2765991834394095,
    #           3.813150712101844, 3.8412788522292276]
    # x_left = 1.0
    # x_right = 1
    
    
    df = pd.DataFrame({'left': left, 'right': right})
    
    
    
    plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 1)
    ax1 = plt.subplot(gs[:, :])
    ax1.plot(df['left'], '-k')
    ax2 = ax1.twinx()
    ax2.plot(df['right'], '-b')
    # Lmin, Lmax = ax1.get_ybound()
    # Rmin, Rmax = ax2.get_ybound()
    # k = (Lmax-Lmin) / (Rmax-Rmin)
    # b = Lmin - k * Rmin
    # x_right_new = k * x_right + b
    # dif = x_right_new - x_left
    # if dif >= 0:
    #     Rmin_new = ((Lmin-dif) - b) / k
    #     k_new = (Lmin-x_left) / (Rmin_new-x_right)
    #     b_new = x_left - k_new * x_right
    #     Rmax_new = (Lmax - b_new) / k_new
    # else:
    #     Rmax_new = ((Lmax-dif) - b) / k
    #     k_new = (Lmax-x_left) / (Rmax_new-x_right)
    #     b_new = x_left - k_new * x_right
    #     Rmin_new = (Lmin - b_new) / k_new    
    # def _inverse(x):
    #     return (x - b_new) / k_new
    # def _forward(x):
    #     return k_new * x + b_new    
    # ax2.set_ylim([Rmin_new, Rmax_new])
    # ax2.set_yscale('function', functions=(_forward, _inverse))
    ax1, ax2 = twinx_align(ax1, ax2, x_left, x_right)
    
    ax1.set_yticks([min(left), x_left, max(left)])
    ax2.set_yticks([min(right), x_right, max(right)])
    # ax1.set_yticks([x_left])
    # # ax2.set_yticks([x_right])
    ax2.set_yticks([5, 10, 15, 20, 25, 30])
    
    ax1.grid(axis='y', c='r', lw='2')
    ax2.grid(axis='y')
    # ax2.grid()
    plt.show()