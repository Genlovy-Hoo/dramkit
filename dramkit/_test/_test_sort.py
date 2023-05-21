# -*- coding: utf-8 -*-

import numpy as np
from dramkit.gentools import func_runtime_test
from dramkit.sorts.insert_sort import insert_sort
from dramkit.sorts.insert_sort_bin import insert_sort_bin
from dramkit.datsci.stats import cumrank, _cumrank1, _cumrank2


if __name__ == '__main__':
    nums = np.random.randint(1, 1000, size=(1000,))
    
    n = 5
    return_all = True
    
    # # 普通插入排序和二分法插入排序性能测试
    # t1, res1 = func_runtime_test(insert_sort, n=n,
    #                              return_all=return_all,
    #                              nums_ori=nums)
    # t2, res2 = func_runtime_test(insert_sort_bin, n=n,
    #                              return_all=return_all,
    #                              nums_ori=nums)
    
    # 累计排序性能测试
    t1, res1 = func_runtime_test(cumrank, n=n,
                                  return_all=return_all,
                                  series=nums)
    t2, res2 = func_runtime_test(_cumrank1, n=n,
                                  return_all=return_all,
                                  series=nums)
    t3, res3 = func_runtime_test(_cumrank2, n=n,
                                  return_all=return_all,
                                  series=nums)
    res3 = [list(x) for x in res3]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    