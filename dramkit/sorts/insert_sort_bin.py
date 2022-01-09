# -*- coding: utf-8 -*-

'''
二分法改进的插入排序法

.. todo::
    待排序数组中存在无效值的处理

References
----------
https://cuijiahua.com/blog/2017/12/algorithm_2.html
'''


import numpy as np
from dramkit.gentools import isnull


def bin_search(nums_sorted, num, start=None, end=None, ascending=True):
    '''
    二分法查找num应该插入到排序好的nums_sorted中的位置

    Parameters
    ----------
    nums_sorted : list
        已经排序好的数据列表
    num :
        待插入数据
    start : None, int
        指定查找开始位置，默认为开头
    end : None, int
        指定查找结束位置，默认为结尾
    ascending : bool
        设置是否为升序排列
        
    Note
    ----
    nums_sorted和num中均不能有无效值
    '''

    start = 0 if start is None else start
    end = len(nums_sorted)-1 if end is None else end
    left, right = start, end

    if ascending:
        while left <= right:
            middle = left + (right - left) // 2
            if nums_sorted[middle] > num:
                right = middle - 1
            else:
                left = middle + 1
    else:
        while left <= right:
                middle = left + (right - left) // 2
                if nums_sorted[middle] < num:
                    right = middle - 1
                else:
                    left = middle + 1

    return left


def insert_sort_bin(nums_ori, ascending=True):
    '''
    二分法改进的插入排序法

    Parameters
    ----------
    nums_ori : list
        待排序数组
    ascending : bool
        是否升序排列


    :returns: `list` - 返回排序之后的数据列表
    '''
    nums = list(nums_ori)
    nums = [x for x in nums if not isnull(x)]
    nums_sorted = [nums[0]]
    for k in range(1, len(nums)):
        num = nums[k]
        insert_iloc = bin_search(nums_sorted, num, ascending=ascending)
        nums_sorted.insert(insert_iloc, num)
    nums_sorted = nums_sorted + [np.nan] * (len(nums_ori)-len(nums_sorted))
    return nums_sorted


def rank_of_insert_bin(nums_sorted, ranks, num,
                       ascending=True, method='dense'):
    '''
    | 在排好序的nums_sorted中插入新元素num，并返回num的排序号，用二分法改进
    | ranks为已经排好序的nums_sorted的序号列表
    | method意义同pandas中rank的method参数
    | 返回顺序：num的排序号, (排好的新序列列表, 排好的新序列号列表)

    Note
    ----
    nums_sorted和num中均不能有无效值
    '''

    if method not in ['average', 'min', 'max', 'first', 'dense']:
        raise ValueError('未识别的并列排序方式！')

    nums_sorted_new = list(nums_sorted)
    k = bin_search(nums_sorted_new, num, ascending=ascending)
    nums_sorted_new.insert(k, num)

    if k == 0:
        irank = 1
        ranks_new = [1] + [x+1 for x in ranks]
        return irank, (nums_sorted_new, ranks_new)

    n = len(nums_sorted_new)
    ranks_new = list(ranks)
    if method == 'first':
        irank = k+1
        ranks_new = list(range(1, n+1))
    elif method == 'dense':
        if num == nums_sorted[k-1]:
            irank = ranks[k-1]
            ranks_new.insert(k, irank)
        else:
            irank = ranks[k-1] + 1
            ranks_new.insert(k, irank)
            for i in range(k+1, n):
                ranks_new[i] = ranks[i-1] + 1
    elif method == 'min':
        if num == nums_sorted[k-1]:
            irank = ranks[k-1]
        else:
            irank = k+1
        ranks_new.insert(k, irank)
        for i in range(k+1, n):
            ranks_new[i] = ranks[i-1] + 1
    elif method == 'max':
        if num == nums_sorted[k-1]:
            irank = ranks[k-1]+1
            j = k-1
            while j > 0 and nums_sorted[j] == num:
                ranks_new[j] += 1
                j -= 1
        else:
            irank = k+1
        ranks_new.insert(k, irank)
        for i in range(k+1, n):
            ranks_new[i] = ranks[i-1] + 1
    elif method == 'average':
        if num == nums_sorted[k-1]:
            irank = ranks[k-1] + 0.5
            j = k-1
            while j > 0 and nums_sorted[j] == num:
                ranks_new[j] += 0.5
                j -= 1
        else:
            irank = k+1
        ranks_new.insert(k, irank)
        for i in range(k+1, n):
            ranks_new[i] = ranks[i-1] + 1

    return irank, (nums_sorted_new, ranks_new)


if __name__ == '__main__':
    nums_ori = [6, 4, 8, 3, np.nan, 9, 2, 3, 1]
    nums_sorted = insert_sort_bin(nums_ori, ascending=True)
    print('排序前:', nums_ori)
    print('排序后:', nums_sorted)
