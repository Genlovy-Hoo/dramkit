# -*- coding: utf-8 -*-

'''
参考:
https://my.oschina.net/u/3242615/blog/1940533/
'''


def backpack01_float_dy(c, v, w):
    '''
    动态规划解决（正）浮点数0-1背包问题

    Parameters
    ----------
    c : float
        背包最大容量
    v : list
        物品价值列表
    w : list
        物品重量列表

    Returns
    -------
    max_v : float
        最大价值
    trace : list
        最大价值解对应物品编号（从0开始计）
    c_v : list
        最大容量-最大价值跳跃点完整记录
        
        Note
        ----
        c_v[_] = [W, V]表示某个物品加入之后，最大容量为W时最大价值为V
    '''

    if len(v) != len(w):
        raise Exception('需要保持v和w长度相等！')

    n = len(v)
    c_v = [[0, 0]]
    # head存放每个物品对应的跳跃点记录在c_v中的开始索引
    head = [None for _ in range(0, n+2)]
    head[n+1] = 0
    head[n] = 1
    # idx_strt|idx_end在每轮迭代中记录上一个物品对应的跳跃点在c_v中的开始|结束索引
    idx_strt = 0
    idx_end = 0

    # 迭代过程：
    # 对上一个物品的每个跳跃点(由j遍历)，假设新物品(由i遍历)加入，新增跳跃点[nw, nv]
    # 新跳跃点[nw, nv]需要与上一个物品的每个跳跃点(由k遍历)作比较，以确定当前物品的跳跃点：
        # 若已有跳跃点重量小于nw，则保留已有跳跃点
        # 若重量相等，则取价值最大的保留或替换
        # 若nw和nv均大于已有跳跃点，则新增[nw, nv]
        # 若已有跳跃点重量大于nw，价值却小于nv，则删除已有跳跃点
        # 若已有跳跃点重量大于nw同时价值大于nv，则保留已有跳跃点
    # 整个比较过程中，k记录了已经与新增跳跃点[nw, nv]比较过的已有跳跃点，从而避免重复比较
    #（由于每次更新之后跳跃点都是按重量和价值升序排列的，因此上一个物品已有的跳跃点只需要与
    # 新加入的跳跃点比较一次即可）
    for i in range(n-1, -1, -1):
        # idx_strt <= k <= idx_end
        # k用于标注上一个物品的跳跃点c_v[k]已经与当前物品比较并处理过（留、换、增、删）
        k = idx_strt
        for j in range(idx_strt, idx_end+1):
            if c_v[j][0] + w[i] > c: # 超过最大容量，不能再加入
                break

            nw = c_v[j][0] + w[i]
            nv = c_v[j][1] + v[i]

            # 保留
            while k <= idx_end and c_v[k][0] < nw:
                c_v.append([c_v[k][0], c_v[k][1]])
                k += 1

            # 替换（也可能保留）或新增
            if k <= idx_end and c_v[k][0] == nw:
                if c_v[k][1] > nv:
                    nv = c_v[k][1]
                k += 1
            if nv > c_v[-1][1]:
                c_v.append([nw, nv])

            # 删除（这里后续的已有跳跃点其重量都大于nw，若其价值小于nv，则删除）
            while k <= idx_end and c_v[k][1] <= nv:
                k += 1

        # 保留（这里后续的已有跳跃点其重量和价值都大于[nw, nv]）
        # 注意这里不能放在遍历j的for循环里，否则可能会导致某些跳跃点被遗漏
        while k <= idx_end:
            c_v.append([c_v[k][0], c_v[k][1]])
            k += 1

        idx_strt = idx_end + 1
        idx_end = len(c_v) - 1
        head[i] = len(c_v)

    max_v = c_v[-1][1]
    trace = _trace_back(v, w, c_v, head)

    return max_v, trace, c_v


def _trace_back(v, w, c_v, head):
    '''
    （正）浮点数0-1背包问题动态规划路径溯源

    Parameters
    ----------
    v : list
        物品价值列表
    w : list
        物品重量列表
    c_v : list
        最大容量-最大价值跳跃点完整记录
    head : list
        存放每个物品对应的跳跃点记录在c_v中的开始索引

    Returns
    -------
    trace : list
        最优路径（最大价值对应物品编号列表，编号从0开始计）
    '''

    trace = []
    k = head[0] - 1
    n = len(w)

    for i in range(1, n+1):
        idx_strt = head[i+1] # i对应物品的跳跃点在c_v中的开始索引
        idx_end = head[i] - 1 # i对应物品的跳跃点在c_v中的结束索引

        for j in range(idx_strt, idx_end + 1):
            if c_v[j][0] + w[i-1] == c_v[k][0] and \
               c_v[j][1] + v[i-1] == c_v[k][1]: # 发生过替换或新装入物品操作
                k = j
                trace.append(i-1)
                break

    return trace


def backpack01_float_dy_smp(c, v, w):
    '''
    动态规划解决（正）浮点数0-1背包问题，只输出最大价值，不考虑路径回溯

    Parameters
    ----------
    c : float
        背包最大容量
    v : list
        物品价值列表
    w : list
        物品重量列表

    Returns
    -------
    max_v : float
        最大价值
    '''

    if len(v) != len(w):
        raise Exception('需要保持v和w长度相等！')

    n = len(v)
    values = [[0, 0]]
    # values为最大容量-最大价值跳跃点记录
    # values[_] = [W, V]表示某个物品加入之后，最大容量为W时最大价值为V

    for i in range(n-1, -1, -1):
        k = 0
        values_new = []
        for j in range(0, len(values)):
            if w[i] + values[j][0] > c:
                break

            nw = w[i] + values[j][0]
            nv = v[i] + values[j][1]

            while k < len(values) and values[k][0] < nw:
                values_new.append(values[k])
                k += 1

            if k < len(values) and values[k][0] == nw:
                if values[k][1] > nv:
                    nv = values[k][1]
                k += 1

            if nv > values_new[-1][1]:
                values_new.append([nw, nv])

            while k < len(values) and values[k][1] < nv:
                k += 1

        while k < len(values):
            values_new.append(values[k])
            k += 1

        values = values_new
        
    max_v = values[-1][1]

    return max_v


if __name__ == '__main__':
    # c = 10
    # v = [6, 3, 5, 4, 6]
    # w = [2, 2, 6, 5, 4]

    c = 9
    v = [1, 1, 7, 6]
    w = [2, 2, 4, 4]

    c = 10.2
    v = [2, 3, 1, 5.2, 4, 3]
    w = [2, 2, 3, 1.2, 5, 2]

    c = 22 + 21 + 8.1
    v = [22, 15, 14, 13, 7, 6.1, 5, 21.5, 100]
    w = [22, 15, 14, 13, 7, 6.1, 5, 21.5, 100]

    max_v = backpack01_float_dy_smp(c, v, w)
    print('最大价值smp:', max_v)

    max_v, trace, c_v = backpack01_float_dy(c, v, w)
    print('最大价值:', max_v)
    print('最大重量:', sum([w[x] for x in trace]))
    print('最大价值对应物品编号:', trace)
    print('最大价值对应物品重量:', [w[x] for x in trace])
    print('最大价值对应物品价值:', [v[x] for x in trace])
