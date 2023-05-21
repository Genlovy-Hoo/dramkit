# -*- coding: utf-8 -*-

import time
import multiprocessing


def task(num):
    time.sleep(num)
    return num * num


if __name__ == '__main__':
    from dramkit.gentools import TimeRecoder
    
    
    # 这样写并没有多进程加速
    tr = TimeRecoder()
    with multiprocessing.Pool(processes=3) as pool:
        results = []
        for i in [10, 5, 20]:
            result = pool.apply_async(task, args=(i,))
            results.append(result.get())
        print(results)
        tr.used()
    
    
    # 这样写有多进程加速
    tr = TimeRecoder()
    pool = multiprocessing.Pool(processes=3)
    args = [10, 5, 20]
    for i in args:
        exec('task_%s = pool.apply_async(task, args=(i,))'%i)
    pool.close()
    pool.join()
    results = [eval('task_%s.get()'%i) for i in args]
    print(results)
    tr.used()
    

