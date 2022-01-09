# -*- coding: utf-8 -*-


def bubble_sort(nums_ori, ascending=True):
    '''
    冒泡排序算法: 比较两个相邻的元素，将值大的元素交换到右边(升序情况)

    Note
    ----
    - 时间复杂度O(n^2)，n为待排序数的个数
    - 缺点：时间复杂度高
    - 优点: 不像桶排序那样只能排正整数

    Parameters
    ----------
    nums_ori : list
        待排序数列表
    ascending : bool
        是否升序，默认True


    :returns: `list` - 返回排序之后的数据列表
    '''

    nums = list(nums_ori)
    n = len(nums)

    for i in range(0, n):
        for j in range(0, n-1-i):
            if ascending:
                if nums[j] > nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
            else:
                if nums[j] < nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums


if __name__ == '__main__':
    nums_ori = [5, 3, 5, 2, 8, -1, 10.3, 9.5]

    nums_sort_asc = bubble_sort(nums_ori)
    nums_sort_des = bubble_sort(nums_ori, ascending=False)

    print('ori nums:')
    print(nums_ori)
    print('ascending sorted nums:')
    print(nums_sort_asc)
    print('descending sorted nums:')
    print(nums_sort_des)
