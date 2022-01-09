# -*- coding: utf-8 -*-

'''
插入排序

.. todo::
    待排序数组中存在无效值的处理
'''


def insert_sort(nums_ori, ascending=True):
    '''
    插入排序法：将未排序的数据依次插入到有序序列中

    .. note::
        时间复杂度最坏为O(n^2)，最好为O(n)，n为待排序数的个数

    Parameters
    ----------
    nums_ori : list
        待排序数组
    ascending : bool
        是否升序排列


    :returns: `list` - 返回排序之后的数据列表

    References
    ----------
    https://blog.csdn.net/weixin_43790276/article/details/104033635
    '''
    nums = list(nums_ori)
    if ascending:
        for i in range(len(nums)):
            j = i
            while j-1 >= 0 and nums[j-1] > nums[j]:
                nums[j], nums[j-1] = nums[j-1], nums[j]
                j -= 1
        return nums
    else:
        for i in range(len(nums)):
            j = i
            while j-1 >= 0 and nums[j-1] < nums[j]:
                nums[j], nums[j-1] = nums[j-1], nums[j]
                j -= 1
        return nums


def insert_to_sorted(nums_sorted, num, ascending=True):
    '''
    | 在已经排序好的nums_sorted (`list`)中插入num并排序
    | 返回加入新数据并排序之后的列表
    '''
    nums_new = list(nums_sorted)
    k = 0
    if ascending:
        while k < len(nums_new) and num >= nums_new[k]:
            k += 1
    else:
        while k < len(nums_new) and num <= nums_new[k]:
            k += 1
    nums_new.insert(k, num)
    return nums_new


def rank_of_insert(nums_sorted, ranks, num,
                   ascending=True, method='dense'):
    '''
    | 在排好序的nums_sorted中插入新元素num，并返回num的排序号
    | ranks为已经排好序的nums_sorted的序号列表
    | method意义同pandas中rank的method参数
    | 返回顺序：num的排序号, (排好的新序列列表, 排好的新序列号列表)
    '''

    if method not in ['average', 'min', 'max', 'first', 'dense']:
        raise ValueError('未识别的并列排序方式！')

    nums_sorted_new = list(nums_sorted)
    k = 0
    if ascending:
        while k < len(nums_sorted_new) and num >= nums_sorted_new[k]:
            k += 1
    else:
        while k < len(nums_sorted_new) and num <= nums_sorted_new[k]:
            k += 1
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
    # nums_ori = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
    nums_ori = [5, 3, 5, 2, 8, -1, 10.3, 9.5]

    nums_sort_asc = insert_sort(nums_ori)
    nums_sort_des = insert_sort(nums_ori, ascending=False)

    print('ori nums:')
    print(nums_ori)
    print('ascending sorted nums:')
    print(nums_sort_asc)
    print('descending sorted nums:')
    print(nums_sort_des)


    nums_sorted = [1, 2, 3, 6, 9][::-1]
    num = 5
    ascending = False
    nums_new = insert_to_sorted(nums_sorted, num, ascending=ascending)
    print(nums_sorted)
    print(nums_new)
