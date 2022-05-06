# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd


def get_1bet_nums(n, Pmax, Pmin=1, **kwargs):
    '''
    随机生成一注号码

    Parameters
    ----------
    n: 一注号码的个数
    Pmax: 备选号码最大值
    Pmin: 备选号码最小值

    Returns
    -------
    bet_nums: 随机选中的一注号码列表
    '''

    # 无放随机回抽样
    bet_nums = np.random.choice(range(Pmin, Pmax+1), size=n, replace=False)

    return bet_nums


def play_Nrounds_6mark_half2(n_bet=5, N=10000, multer=1):
    '''
    六合彩后面部分模拟玩N次
    n_bet: 每次下注的号码个数
    N: 模拟次数
    multer: 倍数
    '''

    gain_total = 0
    for k in range(0, N):
        cost = n_bet * multer # 成本
        get_back = 0

        bet_nums = get_1bet_nums(n_bet, 49) # 投注号码
        hit_num = get_1bet_nums(1, 49)[0] # 中奖号码

        if hit_num in bet_nums:
            get_back = 46 * multer

        gain = get_back - cost
        gain_total += gain

    return gain_total / (n_bet * multer * N)

if __name__ == '__main__':
    from dramkit import plot_series
    
    strt_tm = time.time()

    N = 2000000
    multer = 1

    if not os.path.exists('gains{}.csv'.format(N)):
        gains = []
        for n_bet in range(1, 50):
            print(n_bet, end=' ')
            gain_rate = play_Nrounds_6mark_half2(n_bet, N, multer)
            gains.append([n_bet, gain_rate])
        max_gain = max(gains, key=lambda x: x[1])
        print('\n')
        print('best n_bet: {}'.format(max_gain[0]))
        print('best gain_rate: {}'.format(max_gain[1]))

        gains_pd = pd.DataFrame(gains, columns=['n_bet', 'gain_rate'])
        gains_pd['rank'] = gains_pd['gain_rate'].rank(ascending=False)
    else:
        gains_pd = pd.read_csv('gains{}.csv'.format(N))
        gains_pd['rank'] = gains_pd['gain_rate'].rank(ascending=False)
        gains = gains_pd[['n_bet', 'gain_rate']].values.tolist()
        max_gain = max(gains, key=lambda x: x[1])
        print('\n')
        print('best n_bet: {}'.format(max_gain[0]))
        print('best gain_rate: {}'.format(max_gain[1]))

    gains_pd.set_index('n_bet', inplace=True)
    gains_pd.to_csv('gains{}.csv'.format(N))
    plot_series(gains_pd, {'gain_rate': ('.-b', False)},
                cols_to_label_info={'gain_rate':
                                    [['rank', (1, 2, 3), ('r*', 'm*', 'y*'),
                                      ('最大赢率', '次大赢率', '第三大赢率')]]},
                yparls_info_up=[(gains_pd[gains_pd['rank'] == 1].index[0], 'r', '-', 1.0),
                                (gains_pd[gains_pd['rank'] == 2].index[0], 'm', '-', 1.0),
                                (gains_pd[gains_pd['rank'] == 3].index[0], 'y', '-', 1.0)],
                xparls_info={'gain_rate': [(gains[-1][1], 'k', '-', 1.0)]},
                xlabels=['n_bet'], ylabels=['gain_rate'],
                n_xticks=49, markersize=15, grids=False, figsize=(11, 7))


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
