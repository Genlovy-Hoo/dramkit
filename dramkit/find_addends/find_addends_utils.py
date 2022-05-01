# -*- coding: utf-8 -*-

import numpy as np

#%%
def tol2side_x_eq_y(x, y, tol_below=0.0, tol_above=0.0):
    '''在上界误差tol_above和下界误差tol_below范围内判断x是否等于y'''
    return y - tol_below <= x <= y + tol_above


def tol_eq(x, y, tol=0.0):
    '''在绝对误差tol范围内判断x和y相等'''
    return abs(x - y) <= tol


def tol_x_big_y(x, y, tol=0.0):
    '''在绝对误差tol范围外判断x大于y'''
    return x > y and abs(x - y) > tol


def tol_x_big_eq_y(x, y, tol=0.0):
    '''在绝对误差tol范围内判断x大于等于y'''
    return tol_x_big_y(x, y, tol) or tol_eq(x, y, tol)


def tol_x_sml_y(x, y, tol=0.0):
    '''在绝对误差tol范围外判断x小于y'''
    return x < y and abs(y - x) > tol


def tol_x_sml_eq_y(x, y, tol=0.0):
    '''在绝对误差tol范围内判断x小于等于y'''
    return tol_x_sml_y(x, y, tol) or tol_eq(x, y, tol)

#%%
def get_alts_sml(tgt_sum, alts, sort_type='descend', tol=0.0, add_num=None):
    '''
    从给定备选列表alts中挑选出和小于等于tgt_sum的可行备选数

    Parameters
    ----------
    tgt_sum : float, int
        目标和
    alts : list
        备选数列表
    sort_type : str
        对alts进行排序的方式，默认'descend'降序，可选'ascend'升序、None不排
    tol : float
        两个数进行比较时的绝对误差控制范围
    add_num : int, None
        限制在加起来和大于等于tgt_sum的基础上增加的备选数个数，默认无限制

    Returns
    -------
    alts : list
        可行备选数列表
    '''

    # 备选数不能大于目标和
    alts = [x for x in alts if tol_x_sml_eq_y(x, tgt_sum, tol)]

    if len(alts) == 0:
        return []

    if sort_type == 'descend':
        alts = sorted(alts, reverse=True)
    if sort_type == 'ascend':
        alts = sorted(alts, reverse=False)

    if add_num is None or add_num >= len(alts):
        return alts

    cumSum = list(np.cumsum(alts))
    tmp = [1 if s >= tgt_sum else 0 for s in cumSum]
    try:
        strt_idx = tmp.index(1)
        if strt_idx+add_num+1 <= len(alts):
            return alts[:strt_idx+add_num+1]
        else:
            return alts
    except:
        return alts

#%%
def backfind_sml1st_index(tgt_sum, alts, tol=0.0, loop_count=None):
    '''
    alts从后往前搜索，返回第一个小于等于tgt_sum的数的索引

    Parameters
    ----------
    tgt_sum : int, float
        目标值
    alts : list
        待比较数列表
    tol : float
        两个数进行比较时的绝对误差控制范围
    loop_count : int
        初始迭代次数值，默认为None；若loop_count为None，则不记录迭代次数，
        否则在loop_count基础上继续记录迭代次数

    Returns
    -------
    idx : int
        从后往前搜索，alts中小于等于tgt_sum的第一个数的索引
    loop_count : int
        搜索结束时的迭代次数
    '''
    if len(alts) == 0:
        return -1, loop_count

    idx = len(alts) - 1

    if loop_count is None:
        while idx >= 1 and tol_x_big_y(alts[idx], tgt_sum, tol):
            idx -= 1
        return idx, loop_count
    else:
        while idx >= 1 and tol_x_big_y(alts[idx], tgt_sum, tol):
            idx -= 1
            loop_count += 1
        return idx, loop_count

#%%
if __name__ == '__main__':
    tgt_sum = 10
    alts = [2, 5, 12, 11, 7, 8, 6, 3, 1, 10, 13]


    sort_type = 'descend'
    tol = 1.0
    add_num = None

    alts_new = get_alts_sml(tgt_sum, alts, sort_type=sort_type, tol=tol,
                            add_num=add_num)
    print(alts_new)


    alts = sorted(alts, reverse=False)
    idx, loop_count = backfind_sml1st_index(tgt_sum, alts, tol=tol,
                                            loop_count=None)
    print(alts)
    print(idx, loop_count)
