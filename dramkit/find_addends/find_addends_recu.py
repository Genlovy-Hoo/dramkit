# -*- coding: utf-8 -*-

# from .find_addends_utils import tol2side_x_eq_y
# # from find_addends_utils import tol2side_x_eq_y
from dramkit.find_addends.find_addends_utils import tol2side_x_eq_y


def find_addends_recu(tgt_sum, alts, max_idx, choseds=[], results=[],
                      tol_below=0.0, tol_above=0.0, n_result=None,
                      keep_order=True):
    '''
    | 递归方法，从给定的列表alts中选取若干个数之和最接近tgt_sum
    | 优点: 可在实数范围内求解，没有正负或整数限制；能够返回所有可行解（理论上）
    | 缺点: 受递归深度限制，alts太长就无法求解；没有可行解的情况下不能返回相对最优解？

    Parameters
    ----------
    tgt_sum : float, int
        目标和
    alts : list
        备选数列表
    max_idx : int
        从索引max_idx处开始从后往前搜索，初始值一般设为alts的最大索引
    choseds : list
        预选中的备选数，初始值默认为空
    results : list
        已存在的所有结果，初始值默认为空
    tol_below : float
        下界误差，若目标和减去当前和小于等于tol_below，则为可行解，默认为0.0
    tol_above : tol
        上界误差，若当前和减去目标和小于等于bol_above，则为可行解，默认为0.0
    n_result : int, None
        返回的可行解个数限制，当results的长度大于n_result时返回results
    keep_order : bool
        是否保结果中备选数的进入顺序与alts中的顺序一致，默认是

    Returns
    -------
    results : list
        所有可能的解列表

    References
    ----------
    https://blog.csdn.net/mandagod/article/details/79588753
    '''

    if n_result is None:
        if max_idx < 0:
            return results
    else:
        if len(results) > n_result-1 or max_idx < 0:
            return results

    result = [] # 保存一个可行解
    # 满足误差界限意味着找到一个可行解
    if tol2side_x_eq_y(alts[max_idx], tgt_sum, tol_below, tol_above):
        result.append(alts[max_idx])

        if keep_order:
            choseds = choseds[::-1] # 对choseds逆序（保证顺序）
            result += choseds
            choseds = choseds[::-1] # 恢复顺序
        else:
            result += choseds

        results.append(result)

    # 选中max_idx之后继续递归
    choseds.append(alts[max_idx])
    results = find_addends_recu(tgt_sum-alts[max_idx], alts, max_idx-1,
                               choseds=choseds, results=results,
                               tol_below=tol_below, tol_above=tol_above,
                               n_result=n_result, keep_order=keep_order)

    # 不选max_idx继续递归
    del choseds[-1]
    results = find_addends_recu(tgt_sum, alts, max_idx-1, choseds=choseds,
                               results=results, tol_below=tol_below,
                               tol_above=tol_above, n_result=n_result,
                               keep_order=keep_order)

    return results


if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    tgt_sum = 22 + 21 + 4.1
    alts = [22, 15, 14, 13, 7, 6.1, 5, 21.5, 100]

    # alts = [100, 99, 98, 98.2, 6, 5, 3, -20, -25]
    # tgt_sum = 78.2

    # alts = [200, 107, 100, 99, 98, 6, 5, 3, -1, -20, -25]
    # tgt_sum = 100 + 6 - 25

    # alts = [-900, 901, 800, 600, -400, 402]
    # tgt_sum = 1

    # alts = [100, -100, -105, -102, -25, -30, -1]
    # tgt_sum = -26

    alts = [10, 9, 8, 7, 6, 5]
    tgt_sum = 17

    # alts = [10, 9, 8, -7, -6, -5]
    # tgt_sum = 3

    # alts = [10, 7, 6, 3]
    # tgt_sum = 18

    # alts = [10, 7, 6, 3]
    # tgt_sum = 12


    max_idx = len(alts) - 1
    choseds = []
    results = []
    tol_below = 0.0001
    tol_above = 0.0001
    n_result = None
    keep_order = True

    results = find_addends_recu(tgt_sum, alts, max_idx, choseds=choseds,
                                results=results, tol_below=tol_below,
                                tol_above=tol_above, n_result=n_result,
                                keep_order=keep_order)

    print('可能的序列:')
    for result in results:
        print(result, end='')
        s = sum(result)
        print('\n和:', s, '\n目标和:', tgt_sum, '\n差:', tgt_sum-s, end='\n\n')


    tr.used()
