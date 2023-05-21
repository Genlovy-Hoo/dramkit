# -*- coding: utf-8 -*-

import time
import multiprocessing


def process_data(data):
    time.sleep(data)
    return data**2


if __name__ == '__main__':
    from dramkit.gentools import TimeRecoder
    tr = TimeRecoder()
    
    
    # 创建进程池
    pool = multiprocessing.Pool()

    # 要处理的数据
    data_to_process = [6, 7, 8, 1, 2, 3, 4, 5, 9, 10]

    # 在进程池中处理数据
    processed_data = pool.map(process_data,
                              data_to_process)

    # 关闭进程池
    pool.close()
    pool.join()

    # 处理后的数据结果
    print(processed_data)
    
    
    tr.used()
    
    


