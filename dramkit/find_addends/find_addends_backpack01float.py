# -*- coding: utf-8 -*-


def find_addends_backpack01float(tgt_sum, alts):
    '''
    | 从给定的列表alts（正数）中选取若干个数，其和小于等于tgt_sum并且最接近sum_tgt
    | 思路：求和问题转化为物品价值和重量相等的01（正）浮点数背包问题

    Parameters
    ----------
    tgt_sum : float
        目标和
    alts : list
        备选加数列表

    Returns
    -------
    max_v : float
        最近接tgt_sum的最大和
    addends : list
        和为max_v的备选数列表
    trace : list
        与addends对应的备选加数索引
    values : list
        备选数之和跳跃点记录
        
    References
    ----------
    - https://blog.csdn.net/mandagod/article/details/79588753
    - https://my.oschina.net/u/3242615/blog/1940533/
    '''

    n = len(alts)
    values = [0]
    # head存放每个备选数对应的跳跃点记录在values中的开始索引
    head = [None for _ in range(0, n+2)]
    head[n+1] = 0
    head[n] = 1
    # idx_strt|idx_end在每轮迭代中记录上一个备选数对应的跳跃点在values中的开始|结束索引
    idx_strt = 0
    idx_end = 0

    # 背包求解
    for i in range(n-1, -1, -1):
        k = idx_strt
        for j in range(idx_strt, idx_end+1):
            if values[j] + alts[i] > tgt_sum:
                break

            nS = values[j] + alts[i]

            while k <= idx_end and values[k] < nS:
                values.append(values[k])
                k += 1

            if k <= idx_end and values[k] == nS:
                if values[k] > nS:
                    nS = values[k]
                k += 1
            if nS > values[-1]:
                values.append(nS)

            while k <= idx_end and values[k] <= nS:
                k += 1

        while k <= idx_end:
            values.append(values[k])
            k += 1

        idx_strt = idx_end + 1
        idx_end = len(values) - 1
        head[i] = len(values)

    # 路径回溯
    trace = []
    k = head[0] - 1

    for i in range(1, n+1):
        idx_strt = head[i+1]
        idx_end = head[i] - 1

        for j in range(idx_strt, idx_end + 1):
            if values[j] + alts[i-1] == values[k]:
                k = j
                trace.append(i-1)
                break

    max_v = values[-1]
    addends = [alts[x] for x in trace]

    return max_v, addends, trace, values


if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    tgt_sum = 22 + 21 + 5.1
    alts = [22, 15, 14, 13, 7, 6.1, 5, 21.5, 100]

    # alts = [17541875.0, 16901617.0, 10696518.0, 6388100.0, 2305050.0,
    #         783360.0, 632604.0, 628650.0, 609600.0, 601650.0, 600668.0] + \
    #       [590550.0, 599557.0, 628000.0, 628650.0, 628650.0, 628650.0,
    #         631825.0, 632000.0, 679450.0, 704850.0, 725479.0, 746125.0,
    #         774700.0, 784225.0, 796925.0, 809580.0, 930275.0, 930275.0,
    #         938770.0, 947937.0, 987425.0, 987425.0, 1015625.0, 1054100.0]
    # tgt_sum = 57689692.0 # alts中第一个列表之和

    max_v, addends, trace, values = find_addends_backpack01float(tgt_sum, alts)
    print('目标和:', tgt_sum)
    print('备选加数和:', max_v)
    print('备选加数编号:', trace)
    print('备选加数:', addends)


    tr.used()
