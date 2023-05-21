# -*- coding: utf-8 -*-

from multiprocessing import Process, Queue

from functools import wraps
from dramkit.gentools import isnull
from dramkit.logtools.utils_logger import logger_show


def run_func_with_timeout_process_notwin(
        func, *args, timeout=10, logger_timeout=None,
        timeout_show_str=None, kill_when_timeout=True,
        **kwargs):
    '''
    | 限定时间(timeout秒)执行函数func，若限定时间内未执行完毕，返回None
    | args为tuple或list，为func函数接受的参数列表
    '''
    def _func(res, *args, **kwargs):
        res.put(func(*args, **kwargs))
    res = Queue()
    task = Process(target=_func, args=(res,)+args, kwargs=kwargs)
    task.start() # 启动线程
    task.join(timeout=timeout) # 最大执行时间
    # 超时处理
    if task.is_alive():
        if not isnull(timeout_show_str):
            logger_show(timeout_show_str, logger_timeout, 'warn')
        # 强制结束
        if kill_when_timeout:
            task.terminate()
            # task.join()
    return res.get()


def with_timeout_process_notwin(timeout=30,
                         logger_error=None,
                         logger_timeout=None,
                         timeout_show_str=None,
                         kill_when_timeout=True):
    '''
    | 作为装饰器在指定时间timeout(秒)内运行函数，超时则结束运行
    | 通过控制线程实现

    Examples
    --------
    .. code-block:: python
        :linenos:

        import os
        import pandas as pd
        from dramkit.gentools import tmprint
        
        df1 = pd.DataFrame([[1, 2], [3, 4]])
        df2 = pd.DataFrame([[5, 6], [7, 8]])
        df1.to_excel('df.xlsx')
        TIMEOUT = 3
        
        @with_timeout_process_notwin(TIMEOUT)
        def func(x):
            with open('df.xlsx') as f:
                tmprint('sleeping...')
                time.sleep(5)
            df2.to_excel('df.xlsx')
            return x
        
        def test():
            res = func('test')
            print('res:', res)
            os.remove('df.xlsx')
            
    >>> test()
    '''
    def transfunc(func):
        @wraps(func)
        def timeouter(*args, **kwargs):
            '''尝试在指定时间内运行func，超时则结束运行'''
            return run_func_with_timeout_process_notwin(
                    func, *args, timeout=timeout,
                    logger_timeout=logger_timeout,
                    timeout_show_str=timeout_show_str,
                    kill_when_timeout=kill_when_timeout,
                    **kwargs)
        return timeouter
    return transfunc


def func(x):
    with open('df.xlsx') as f:
        tmprint('sleeping...')
        time.sleep(5)
    df2.to_excel('df.xlsx')
    return x

@with_timeout_process_notwin(3)
def _func(q, *args, **kwargs):
    q.put(func(*args, **kwargs))
    
    
def test():
    res = _func('test')
    print('res:', res)
    os.remove('df.xlsx')
    return res


if __name__ == '__main__':
    import os
    import time
    import pandas as pd
    from dramkit.gentools import tmprint
    
    df1 = pd.DataFrame([[1, 2], [3, 4]])
    df2 = pd.DataFrame([[5, 6], [7, 8]])
    df1.to_excel('df.xlsx')
        
    res = test()







