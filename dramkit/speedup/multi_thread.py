# -*- coding: utf-8 -*-

import inspect, ctypes
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dramkit.logtools.utils_logger import logger_show

#%%
class SingleThread(Thread):
    '''
    | 单个线程任务
    | 参考：
    | https://www.csdn.net/tags/MtTaQg3sMDE1MzYtYmxvZwO0O0OO0O0O.html
    | https://www.cnblogs.com/ojbk6943/p/14047952.html
    | https://blog.csdn.net/weixin_43285186/article/details/124338274
    | https://m.php.cn/article/471342.html
    '''

    def __init__(self, func, args, logger=None):
        '''
        Parameters
        ----------
        func : function
            需要多线程运行的目标函数
        args : list
            目标函数func输入变量列表
        logger : Logger, None
            日志记录器
        '''
        super(SingleThread, self).__init__()
        self.func = func
        self.args = args
        self.logger = logger

    def run(self):
        '''执行目标函数func，获取返回结果'''
        self.result = self.func(*self.args)

    def get_result(self):
        '''获取执行结果'''
        try:
            return self.result
        except:
            logger_show('error occurred, return None.',
                        self.logger, 'error')
            return None
 
    def stop_thread(self):
        '''结束线程'''
        def _async_raise(tid, exctype):
            '''raises the exception, performs cleanup if needed'''
            if not inspect.isclass(exctype):
                exctype = type(exctype)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                   tid, ctypes.py_object(exctype))
            if res == 0:
                raise ValueError('invalid thread id')
            elif res != 1:
                '''
                if it returns a number greater than one, you're in trouble,
                and you should call it again with exc=NULL to revert the effect
                '''
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                raise SystemError('PyThreadState_SetAsyncExc failed')
        _async_raise(self.ident, SystemExit)


def multi_thread_threading(func, args_list, logger=None):
    '''
    多线程，同一个函数执行多次

    Parameters
    ----------
    func : function
        需要多线程运行的目标函数
    args_list : list
        每个元素都是目标函数func的参数列表
    logger : Logger
        logging库的日志记录器

    Returns
    -------
    results : list
        每个元素对应func以args_list的元素为输入的返回结果
    '''

    tasks = []
    for args in args_list:
        task = SingleThread(func, args, logger=logger)
        tasks.append(task)
        task.start()

    results = []
    for task in tasks:
        task.join()
        results.append(task.get_result())

    return results

#%%
def multi_thread_concurrent(func, args_list,
                            multi_line=None, keep_order=True):
    '''
    多线程，同一个函数执行多次

    Parameters
    ----------
    func : function
        需要多线程运行的目标函数
    args_list : list
        每个元素都是目标函数func的参数列表
    multi_line : int, None
        最大线程数，默认等于len(args_list)
    keep_order : bool
        是否保持输入args_list与输出results参数顺序一致性，默认是

    Returns
    -------
    results : list
        每个元素对应func以args_list的元素为输入的返回结果
    '''

    if multi_line is None:
        multi_line = len(args_list)

    # submit方法不能保证results的值顺序与args_list一一对应
    if not keep_order:
        with ThreadPoolExecutor(max_workers=multi_line) as executor:
            futures = [executor.submit(func, *args) for args in args_list]

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # 使用map可保证results的值顺序与args_list一一对应
    if keep_order:
        def func_new(args):
            return func(*args)

        with ThreadPoolExecutor(max_workers=multi_line) as executor:
            results = executor.map(func_new, args_list)
            results = list(results)

    return results

#%%
if __name__ == '__main__':
    import time


    def func(idx, sleep_tm):
        print('task id:', idx)
        time.sleep(sleep_tm)
        print('task id: {}; slept: {}s.'.format(idx, sleep_tm))
        return [idx, sleep_tm]

    args_list = [[1, 2], [3, 4], [4, 5], [2, 3]]


    print('multi-thread, threading..............................')
    strt_tm = time.time()
    results_threading = multi_thread_threading(func, args_list)
    print('used time: {tm}s.'.format(tm=round(time.time() - strt_tm,6)))


    print('multi-thread, concurrent.............................')
    strt_tm = time.time()
    results_concurrent_Order = multi_thread_concurrent(func, args_list,
                                                     keep_order=True)
    results_concurrent_noOrder = multi_thread_concurrent(func, args_list,
                                                       keep_order=False)
    print('used time: {tm}s.'.format(tm=round(time.time() - strt_tm,6)))
