# -*- coding: utf-8 -*-

import time
import multiprocessing


def add(a, b, result_queue):
    time.sleep(a)
    result_queue.put(a + b)
    

def subtract(a, b, result_queue):
    time.sleep(a)
    result_queue.put(a - b)
    

def multiply(a, b, result_queue):
    time.sleep(a)
    result_queue.put(a * b)
    

if __name__ == '__main__':
    from dramkit.gentools import TimeRecoder
    tr = TimeRecoder()
    
    
    result_queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(
         target=add, args=(5, 10, result_queue))
    p2 = multiprocessing.Process(
         target=subtract, args=(15, 5, result_queue))
    p3 = multiprocessing.Process(
         target=multiply, args=(2, 7, result_queue))
    p4 = multiprocessing.Process(
         target=multiply, args=(1, 7, result_queue))

    processes = [p1, p2, p3, p4]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    # 返回结果顺序不对应（结果是按执行完成的顺序排列的）
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    print(results)
    
    
    tr.used()

