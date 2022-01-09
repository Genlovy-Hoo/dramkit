# -*- coding: utf-8 -*-


def quick_sort(nums_ori, ascending=True):
    '''
    快速排序算法: 取一个基数，大于和小于基数的书分别放两边，两边再递归

    Note
    ----
    时间复杂度O(N*logN)，N为待排序数的个数

    Parameters
    ----------
    nums_ori : list
        待排序数列表
    ascending : bool
        是否升序，默认True


    :returns: `list` - 返回排序之后的数据列表
    '''

    nums = list(nums_ori)
    N = len(nums)

    if N <= 1:
        return nums
    else:
        pivot = nums.pop() # 取最后一个作为基数
        bigs, smls = [], []

        for k in nums:
            if k > pivot:
                bigs.append(k)
            else:
                smls.append(k)

        if ascending:
            return quick_sort(smls) + [pivot] + quick_sort(bigs)
        else:
            return quick_sort(bigs, False) + [pivot] + quick_sort(smls, False)


if __name__ == "__main__":
    nums_ori = [5, 3, 5, 2, 8, -1, 10.3, 9.5]

    nums_sort_asc = quick_sort(nums_ori)
    nums_sort_des = quick_sort(nums_ori, ascending=False)

    print('ori nums:')
    print(nums_ori)
    print('ascending sorted nums:')
    print(nums_sort_asc)
    print('descending sorted nums:')
    print(nums_sort_des)
