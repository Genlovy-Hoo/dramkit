# -*- coding: utf-8 -*-


def bucket_sort(nums, ascending=True):
    '''
    桶排序算法

    Note
    ----
    - 时间复杂度O(m+n)，m为桶的个数(nums的最大值+1)，n为待排序数的个数
    - 缺点：

        * 需要事先知道最大最小值
        * 浪费空间（桶的个数受最大值影响极大）
        * 小数点和负数处理麻烦(可通过转化成正整数再排序)
    - 优点: 时间复杂度低

    Parameters
    ----------
    nums : list
        待排序数列表
    ascending : bool
        是否升序，默认True


    :returns: `list` - 返回排序之后的数据列表
    '''

    m = max(nums) + 1
    bucket = [0 for k in range(0, m)]

    for k in nums:
        bucket[k] += 1

    nums_sort = []
    if ascending:
        for i in range(0, m):
            for j in range(0, bucket[i]):
                nums_sort.append(i)
    else:
        for i in range(m-1, -1, -1):
            for j in range(0, bucket[i]):
                nums_sort.append(i)

    return nums_sort


if __name__ == '__main__':
    nums = [5, 3, 5, 2, 8, 1, 10, 9, 3, 5]

    nums_sort_asc = bucket_sort(nums)
    nums_sort_des = bucket_sort(nums, ascending=False)

    print('ori nums:')
    print(nums)
    print('ascending sorted nums:')
    print(nums_sort_asc)
    print('descending sorted nums:')
    print(nums_sort_des)
