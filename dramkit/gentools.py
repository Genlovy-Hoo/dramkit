# -*- coding: utf-8 -*-

'''
General toolboxs
'''

import sys
import time
import copy
import inspect
import numpy as np
import pandas as pd
from functools import reduce, wraps
from random import randint, random, uniform
from dramkit.logtools.utils_logger import logger_show
from dramkit.speedup.multi_thread import SingleThread


PYTHON_VERSION = float(sys.version[:3])


class StructureObject(object):
    '''类似于MATLAB结构数组，存放变量，便于直接赋值和查看'''

    def __init__(self, dirt_modify=True, **kwargs):
        '''初始化'''
        self.set_dirt_modify(dirt_modify)
        self.set_from_dict(kwargs)
        
    @property
    def dirt_modify(self):
        return self.__dirt_modify
        
    def set_dirt_modify(self, dirt_modify):
        assert isinstance(dirt_modify, bool)
        self.__dirt_modify = dirt_modify
        
    def __setattr__(self, key, value):
        _defaults = ['__dirt_modify']
        _defaults = ['_StructureObject' + x for x in _defaults]
        if key in _defaults:
            self.__dict__[key] = value
            return
        if self.dirt_modify:
            self.__dict__[key] = value
        else:
            raise DirtModifyError('不允许直接赋值！')
            # raise DirtModifyError(
            #     '不允许直接赋值，请调用`set_key_value`或`set_from_dict`方法！')

    def __repr__(self):
        '''查看时以key: value格式打印'''
        _defaults = ['__dirt_modify']
        _defaults = ['_StructureObject' + x for x in _defaults]
        return ''.join('{}: {}\n'.format(k, v) for k, v in self.__dict__.items() \
                       if k not in _defaults)
            
    @property
    def keys(self):
        '''显示所有key'''
        _defaults = ['__dirt_modify']
        _defaults = ['_StructureObject' + x for x in _defaults]
        return [x for x in self.__dict__.keys() if x not in _defaults]
    
    @property
    def items(self):
        _defaults = ['__dirt_modify']
        _defaults = ['_StructureObject' + x for x in _defaults]
        for x in self.__dict__.keys():
            if not x in _defaults:
                d = (x, eval('self.{}'.format(x)))
                yield d
    
    def set_key_value(self, key, value):
        self.__dict__[key] = value
    
    def set_from_dict(self, d):
        '''通过dict批量增加属性变量'''
        assert isinstance(d, dict), '必须为dict格式'
        self.__dict__.update(d)
        
    def merge(self, o):
        '''从另一个对象中合并属性和值'''
        assert isinstance(o, (list, tuple, type(self)))
        if isinstance(o, type(self)):
            o = [o]
        for x in o:
            for key in x.keys:
                exec('self.set_key_value(key, x.{})'.format(key))
    
    def copy(self):
        return copy.deepcopy(self)
    
    def pop(self, key):
        '''删除属性变量key，有返回'''
        return self.__dict__.pop(key)
    
    def remove(self, key):
        '''删除属性变量key，无返回'''
        del self.__dict__[key]
    
    def clear(self):
        '''清空所有属性变量'''
        self.__dict__.clear()
        
        
class DirtModifyError(Exception):
    pass
        
        
def run_func_with_timeout(func, args, timeout=10):
    '''
    | 限定时间(timeout秒)执行函数func，若限定时间内未执行完毕，返回None
    | args为tuple或list，为func函数接受的参数列表
    '''
    task = SingleThread(func, args, False) # 创建线程
    task.start() # 启动线程
    task.join(timeout=timeout) # 最大执行时间

    # 若超时后，线程依旧运行，则强制结束
    if task.is_alive():
        task.stop_thread()

    return task.get_result()
        
        
def get_func_arg_names(func):
    '''获取函数func的参数名称列表'''
    return inspect.getfullargspec(func).args
        
        
def try_repeat_run(n_max_run=3, logger=None, sleep_seconds=0,
                   force_rep=False):
    '''
    | 作为装饰器尝试多次运行指定函数
    | 使用场景：第一次运行可能出错，需要再次运行(比如取数据时第一次可能连接超时，需要再取一次)
    
    Parameters
    ----------
    n_max_run : int
        最多尝试运行次数
    logger : None, logging.Logger
        日志记录器
    sleep_seconds : int, float
        | 多次执行时，上一次执行完成之后需要暂停的时间（秒）
        | 注：在force_rep为True时，每次执行完都会暂停，force_rep为False只有报错之后才会暂停
    force_rep : bool
        若为True，则不论是否报错都强制重复执行，若为False，则只有报错才会重复执行
        
    Returns
    -------
    result : None, other
        若目标函数执行成功，则返回执行结果；若失败，则返回None

    Examples
    --------
    .. code-block:: python
        :linenos:

        from dramkit import simple_logger
        logger = simple_logger()
        
        @try_repeat_run(2, logger=logger, sleep_seconds=0, force_rep=False)
        def rand_div(x):
            return x / np.random.randint(-1, 1)

        def repeat_test(info_):
            print(info_)
            return rand_div(0)

    >>> a = repeat_test('repeat test...')
    >>> print(a)
    '''
    def transfunc(func):
        @wraps(func)
        def repeater(*args, **kwargs):
            '''尝试多次运行func'''
            if not force_rep:
                n_run, ok = 0, False
                while not ok and n_run < n_max_run:
                    n_run += 1
                    # logger_show('第%s次运行`%s`...'%(n_run, func.__name__),
                    #             logger, 'info')
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except:
                        if n_run == n_max_run:
                            logger_show('`%s`运行失败，共运行了%s次。'%(func.__name__, n_run),
                                        logger, 'error')
                            return
                        else:
                            if sleep_seconds > 0:
                                time.sleep(sleep_seconds)
                            else:
                                pass
            else:
                n_run = 0
                while n_run < n_max_run:
                    n_run += 1
                    try:
                        result = func(*args, **kwargs)
                        logger_show('`%s`第%s运行：成功。'%(func.__name__, n_run),
                                    logger, 'info')
                    except:
                        logger_show('`%s`第%s运行：失败。'%(func.__name__, n_run),
                                    logger, 'error')
                        result = None
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                return result
        return repeater
    return transfunc


def log_used_time(logger=None):
    '''
    作为装饰器记录函数运行用时
    
    Parameters
    ----------
    logger : None, logging.Logger
        日志记录器

    Examples
    --------
    .. code-block:: python
        :linenos:

        from dramkit import simple_logger
        logger = simple_logger()

        @log_used_time(logger)
        def wait():
            print('wait...')
            time.sleep(3)

    >>> wait()
    wait...
    2021-12-28 12:39:54,013 -utils_logger.py[line: 32] -INFO:
        --function `wait` run time: 3.000383s.

    See Also
    --------
    :func:`dramkit.gentools.print_used_time`

    References
    ----------
    - https://www.cnblogs.com/xiuyou/p/11283512.html
    - https://www.cnblogs.com/slysky/p/9777424.html
    - https://www.cnblogs.com/zhzhang/p/11375574.html
    - https://www.cnblogs.com/zhzhang/p/11375774.html
    - https://blog.csdn.net/weixin_33711647/article/details/92549215
    '''
    def transfunc(func):
        @wraps(func)
        def timer(*args, **kwargs):
            '''运行func并记录用时'''
            t0 = time.time()
            result = func(*args, **kwargs)
            t = time.time()
            logger_show('function `%s` run time: %ss.'%(func.__name__, round(t-t0, 6)),
                        logger, 'info')
            return result
        return timer
    return transfunc


def print_used_time(func):
    '''
    作为装饰器打印函数运行用时
    
    Parameters
    ----------
    func : function
        需要记录运行用时的函数
    
    Examples
    --------
    .. code-block:: python
        :linenos:

        @print_used_time
        def wait():
            print('wait...')
            time.sleep(3)

    >>> wait()
    wait...
    function `wait` run time: 3.008314s.

    See Also
    --------
    :func:`dramkit.gentools.log_used_time`

    References
    ----------
    - https://www.cnblogs.com/slysky/p/9777424.html
    - https://www.cnblogs.com/zhzhang/p/11375574.html
    '''
    @wraps(func)
    def timer(*args, **kwargs):
        '''运行func并print用时'''
        t0 = time.time()
        result = func(*args, **kwargs)
        t = time.time()
        print('function `%s` run time: %ss.'%(func.__name__, round(t-t0, 6)))
        return result
    return timer


def get_update_kwargs(key, arg, kwargs, arg_default=None,
                      func_update=None):
    '''
    取出并更新kwargs中key参数的值

    使用场景：当一个函数接受**kwargs参数，同时需要对kwargs里面的某个key的值进行更新并且
    提取出来单独传参

    Parameters
    ----------
    key :
        kwargs中待取出并更新的关键字
    arg :
        关键字key对应的新值
    kwargs : dict
        关键字参数对
    arg_default : key关键词对应参数默认值
    func_update : None, False, function
        自定义取出的key参数值更新函数: arg_new = func_update(arg, arg_old)

        - 若为False, 则不更新，直接取出原来key对应的参数值或默认值
        - 若为`replace`, 则直接替换更新
        - 若为None, 则 **默认** 更新方式为:

            * 若参数值arg为dict或list类型，则增量更新
            * 若参数值arg_old为list且arg不为list，则增量更新
            * 其它情况直接替换更新

    Returns
    -------
    arg_new :
        取出并更新之后的关键字key对应参数值
    kwargs :
        删除key之后的kwargs

    Examples
    --------
    >>> key, arg = 'a', 'aa'
    >>> kwargs = {'a': 'a', 'b': 'b'}
    >>> get_update_kwargs(key, arg, kwargs)
    ('aa', {'b': 'b'})
    >>> key, arg = 'a', {'a_': 'aa__'}
    >>> kwargs = {'a': {'a': 'aa'}, 'b': 'b'}
    >>> get_update_kwargs(key, arg, kwargs)
    ({'a': 'aa', 'a_': 'aa__'}, {'b': 'b'})
    >>> key, arg = 'a', ['aa', 'aa_']
    >>> kwargs = {'a': ['a'], 'b': 'b'}
    >>> get_update_kwargs(key, arg, kwargs)
    (['a', 'aa', 'aa_'], {'b': 'b'})
    >>> key, arg = 'a', ['aa', 'aa_']
    >>> kwargs = {'a': ['a'], 'b': 'b'}
    >>> get_update_kwargs(key, arg, kwargs, func_update='replace')
    (['aa', 'aa_'], {'b': 'b'})
    >>> key, arg = 'a', 'aa_'
    >>> kwargs = {'a': ['a'], 'b': 'b'}
    >>> get_update_kwargs(key, arg, kwargs)
    (['a', 'aa_'], {'b': 'b'})
    '''

    def _default_update(arg, arg_old):
        if isinstance(arg, dict):
            assert isinstance(arg_old, dict) or isnull(arg_old)
            arg_new = {} if isnull(arg_old) else arg_old
            arg_new.update(arg)
        elif isinstance(arg, list):
            assert isinstance(arg_old, list) or isnull(arg_old)
            arg_new = [] if isnull(arg_old) else arg_old
            arg_new += arg
        elif isinstance(arg_old, list) and not isinstance(arg, list):
            arg_new = arg_old + [arg]
        else:
            arg_new = arg
        return arg_new

    # 取出原来的值
    if key in kwargs.keys():
        arg_old = kwargs[key]
        del kwargs[key]
    else:
        arg_old = arg_default

    # 不更新
    if func_update is False:
        return arg_old, kwargs

    # 更新
    if func_update is None:
        func_update = _default_update
    elif func_update == 'replace':
        func_update = lambda arg, arg_old: arg
    arg_new = func_update(arg, arg_old)
        
    return arg_new, kwargs


def roulette_base(fitness):
    '''
    基本轮盘赌法
    
    Parameters
    ----------
    fitness : list
        所有备选对象的fitness值列表

        .. note::
            fitness的元素值应为正，且fitness值越大，被选中概率越大


    :returns: `int` - 返回被选中对象的索引号

    References
    ----------
    https://blog.csdn.net/armwangEric/article/details/50775206
    '''
    sum_fits = sum(fitness)
    rand_point = uniform(0, sum_fits)
    accumulator = 0.0
    for idx, fitn in enumerate(fitness):
        accumulator += fitn
        if accumulator >= rand_point:
            return idx


def roulette_stochastic_accept(fitness):
    '''
    轮盘赌法，随机接受法
    
    Parameters
    ----------
    fitness : list
        所有备选对象的fitness值列表

        .. note::
            fitness的元素值应为正，且fitness值越大，被选中概率越大


    :returns: `int` - 返回被选中对象的索引号

    References
    ----------
    https://blog.csdn.net/armwangEric/article/details/50775206
    '''
    n = len(fitness)
    max_fit = max(fitness)
    while True:
        idx = randint(0, n-1)
        if random() <= fitness[idx] / max_fit:
            return idx


def roulette_count(fitness, n=10000, rand_func=None):
    '''
    轮盘赌法n次模拟，返回每个备选对象在n次模拟中被选中的次数

    Parameters
    ----------
    fitness : list, dict
        所有备选对象的fitness值列表或字典，格式参见Example
        
        .. note::
            fitness的元素值应为正，且fitness值越大，被选中概率越大
    n : int
        模拟次数
    rand_func : None, function
        | 指定轮盘赌法函数，如'roulette_base'(:func:`dramkit.gentools.roulette_base`)
        | 或'roulette_stochastic_accept'(:func:`dramkit.gentools.roulette_stochastic_accept`),
        | 默认用'roulette_stochastic_accept'
        

    :returns: `list, dict` - 返回每个对象在模拟n次中被选中的次数

    Examples
    --------
    >>> fitness = [1, 2, 3]
    >>> roulette_count(fitness, n=6000)
    [(0, 991), (1, 2022), (2, 2987)]
    >>> fitness = (1, 2, 3)
    >>> roulette_count(fitness, n=6000)
    [(0, 1003), (1, 1991), (2, 3006)]
    >>> fitness = [('a', 1), ('b', 2), ('c', 3)]
    >>> roulette_count(fitness, n=6000)
    [('a', 997), ('b', 1989), ('c', 3014)]
    >>> fitness = [['a', 1], ['b', 2], ['c', 3]]
    >>> roulette_count(fitness, n=6000)
    [('a', 1033), ('b', 1967), ('c', 3000)]
    >>> fitness = {'a': 1, 'b': 2, 'c': 3}
    >>> roulette_count(fitness, n=6000)
    {'a': 988, 'b': 1971, 'c': 3041}
    '''
    
    if rand_func is None:
        rand_func = roulette_stochastic_accept
    
    if isinstance(fitness, dict):
        keys, vals = [], []
        for k, v in fitness.items():
            keys.append(k)
            vals.append(v)
        randpicks = [rand_func(vals) for _ in range(n)]
        idx_picks = [(x, randpicks.count(x)) for x in range(len(vals))]
        return {keys[x[0]]: x[1] for x in idx_picks}

    elif (isinstance(fitness[0], list) or isinstance(fitness[0], tuple)):
        keys, vals = [], []
        for k, v in fitness:
            keys.append(k)
            vals.append(v)
        randpicks = [rand_func(vals) for _ in range(n)]
        idx_picks = [(x, randpicks.count(x)) for x in range(len(vals))]
        return [(keys[x[0]], x[1]) for x in idx_picks]

    elif (isinstance(fitness[0], int) or isinstance(fitness[0], float)):
        randpicks = [rand_func(fitness) for _ in range(n)]
        idx_picks = [(x, randpicks.count(x)) for x in range(len(fitness))]
        return idx_picks


def rand_sum(target_sum, n, lowests, highests, isint=True, n_dot=6):
    '''
    在指定最大最小值范围内随机选取若干个随机数，所选取数之和为定值

    Parameters
    ----------
    target_sum : int, float
        目标和
    n : int
        随机选取个数
    lowests : int, floot, list
        随机数下限值，若为list，则其第k个元素对应第k个随机数的下限
    highests : int, floot, list
        随机数上限值，若为list，则其第k关元素对应第k个随机数的上限
    isint : bool
        所选数是否强制为整数，若为False，则为实数
        
        .. note::
            若输入lowests或highests不是int，则isint为True无效
    n_dot : int
        动态上下界值与上下限比较时控制小数位数(为了避免python精度问题导致的报错)


    :returns: `list` - 随机选取的n个数，其和为target_sum

    Examples
    --------
    >>> rand_sum(100, 2, [20, 30], 100)
    [65, 35]
    >>> rand_sum(100, 2, 20, 100)
    [41, 59]
    >>> rand_sum(100, 2, [20, 10], [100, 90])
    [73, 27]
    '''

    if not (isinstance(lowests, int) or isinstance(lowests, float)): 
        if len(lowests) != n:
            raise ValueError('下限值列表(数组)lowests长度必须与n相等！')
    if  not (isinstance(highests, int) or isinstance(highests, float)):
        if len(highests) != n:
            raise ValueError('上限值列表(数组)highests长度必须与n相等！')

    # lowests、highests组织成list
    if isinstance(lowests, int) or isinstance(lowests, float):
        lowests = [lowests] * n
    if isinstance(highests, int) or isinstance(highests, float):
        highests = [highests] * n

    if any([isinstance(x, float) for x in lowests]) or any([isinstance(x, float) for x in highests]):
        isint = False

    LowHigh = list(zip(lowests, highests))

    def _dyLowHigh(tgt_sum, low_high, n_dot=6):
        '''
        动态计算下界和上界
        tgt_sum为目标和，low_high为上下界对组成的列表
        n_dot为小数保留位数(为了避免python精度问题导致的报错)
        '''
        restSumHigh = sum([x[1] for x in low_high[1:]])
        restSumLow = sum([x[0] for x in low_high[1:]])
        low = max(tgt_sum-restSumHigh, low_high[0][0])
        if round(low, n_dot) > low_high[0][1]:
            raise ValueError(
               '下界({})超过最大值上限({})！'.format(low, low_high[0][1]))
        high = min(tgt_sum-restSumLow, low_high[0][1])
        if round(high, n_dot) < low_high[0][0]:
            raise ValueError(
               '上界({})超过最小值下限({})！'.format(high, low_high[0][0]))
        return low, high

    S = 0
    adds = []
    low, high = _dyLowHigh(target_sum, LowHigh, n_dot=n_dot)
    while len(adds) < n-1:
        # 每次随机选择一个数
        if isint:
            randV = randint(low, high)
        else:
            randV = random() * (high-low) + low

        # 判断当前所选择的备选数是否符合条件，若符合则加入备选数，
        # 若不符合则删除所有备选数重头开始
        restSum = target_sum - (S + randV)
        restSumLow = sum([x[0] for x in LowHigh[len(adds)+1:]])
        restSumHigh = sum([x[1] for x in LowHigh[len(adds)+1:]])
        if restSumLow <= restSum <= restSumHigh:
            S += randV
            adds.append(randV)
            low, high = _dyLowHigh(target_sum-S, LowHigh[len(adds):],
                                  n_dot=n_dot)
        else:
            S = 0
            adds = []
            low, high = _dyLowHigh(target_sum, LowHigh, n_dot=n_dot)

    adds.append(target_sum-sum(adds)) # 最后一个备选数

    return adds


def rand_weight_sum(weight_sum, n, lowests, highests, weights=None, n_dot=6):
    '''
    在指定最大最小值范围内随机选取若干个随机数，所选取数之加权和为定值

    Parameters
    ----------
    weight_sum : float
        目标加权和
    n : int
        随机选取个数
    lowests : int, floot, list
        随机数下限值，若为list，则其第k个元素对应第k个随机数的下限
    highests : int, floot, list
        随机数上限值，若为list，则其第k关元素对应第k个随机数的上限
    weights : None, list
        权重列表
        
        .. note::
            lowests和highests与weights应一一对应
    n_dot : int
        动态上下界值与上下限比较时控制小数位数(为了避免python精度问题导致的报错)


    :returns: `list` - 随机选取的n个数，其以weights为权重的加权和为weight_sum

    Examples
    --------
    >>> rand_weight_sum(60, 2, [20, 30], 100)
    [21.41082008017613, 98.58917991982386]
    >>> rand_weight_sum(70, 2, 20, 100)
    [56.867261610484356, 83.13273838951565]
    >>> rand_weight_sum(80, 2, [20, 10], [100, 90])
    [80.32071140116187, 79.67928859883813]
    >>> rand_weight_sum(80, 2, [20, 10], [100, 90], [0.6, 0.4])
    [88.70409567475888, 66.94385648786168]
    >>> rand_weight_sum(180, 2, [20, 10], [100, 90], [3, 2])
    [23.080418085462018, 55.37937287180697]
    '''

    if weights is not None and len(weights) != n:
        raise ValueError('权重列表W的长度必须等于n！')
    if not (isinstance(lowests, int) or isinstance(lowests, float)):
        if len(lowests) != n:
            raise ValueError('下限值列表(数组)lowests长度必须与n相等！')
    if not (isinstance(highests, int) or isinstance(highests, float)):
        if len(highests) != n:
            raise ValueError('上限值列表(数组)highests长度必须与n相等！')

    # weights和lowests、highests组织成list
    if weights is None:
        weights = [1/n] * n
    if isinstance(lowests, int) or isinstance(lowests, float):
        lowests = [lowests] * n
    if isinstance(highests, int) or isinstance(highests, float):
        highests = [highests] * n

    WLowHigh = list(zip(weights, lowests, highests))

    def _dyLowHigh(wt_sum, w_low_high, n_dot=6):
        '''
        动态计算下界和上界
        wt_sum为目标加权和，w_low_high为权重和上下界三元组组成的列表
        n_dot为小数保留位数(为了避免python精度问题导致的报错)
        '''
        restSumHigh = sum([x[2]*x[0] for x in w_low_high[1:]])
        restSumLow = sum([x[1]*x[0] for x in w_low_high[1:]])
        low = max((wt_sum-restSumHigh) / w_low_high[0][0], w_low_high[0][1])
        if round(low, n_dot) > w_low_high[0][2]:
            raise ValueError(
               '下界({})超过最大值上限({})！'.format(low, w_low_high[0][2]))
        high = min((wt_sum-restSumLow) / w_low_high[0][0], w_low_high[0][2])
        if round(high, n_dot) < w_low_high[0][1]:
            raise ValueError(
               '上界({})超过最小值下限({})！'.format(high, w_low_high[0][1]))
        return low, high

    S = 0
    adds = []
    low, high = _dyLowHigh(weight_sum, WLowHigh, n_dot=n_dot)
    while len(adds) < n-1:
        # 每次随机选择一个数
        randV = random() * (high-low) + low

        # 判断当前所选择的备选数是否符合条件，若符合则加入备选数，
        # 若不符合则删除所有备选数重头开始
        restSum = weight_sum - (S + randV * weights[len(adds)])
        restSumLow = sum([x[1]*x[0] for x in WLowHigh[len(adds)+1:]])
        restSumHigh = sum([x[2]*x[0] for x in WLowHigh[len(adds)+1:]])
        if restSumLow <= restSum <= restSumHigh:
            S += randV * weights[len(adds)]
            adds.append(randV)
            low, high = _dyLowHigh(weight_sum-S, WLowHigh[len(adds):],
                                  n_dot=n_dot)
        else:
            S = 0
            adds = []
            low, high = _dyLowHigh(weight_sum, WLowHigh, n_dot=n_dot)

    aw = zip(adds, weights[:-1])
    adds.append((weight_sum-sum([a*w for a, w in aw])) / weights[-1])

    return adds


def replace_repeat_iter(series, val, val0, gap=None, keep_last=False):
    '''
    替换序列中重复出现的值
    
    | series (`pd.Series`) 中若步长为gap的范围内出现多个val值，则只保留第一条记录，
      后面的替换为val0
    | 若gap为None，则将连续出现的val值只保留第一个，其余替换为val0(这里连续出现val是指
      不出现除了val和val0之外的其他值)
    | 若keep_last为True，则连续的保留最后一个
    
    返回结果为替换之后的series (`pd.Series`)

    Examples
    --------
    >>> data = pd.DataFrame([0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1, 1, 0, 0,
    ...                      -1, -1, 0, 0, 1], columns=['test'])
    >>> data['test_rep'] = replace_repeat_iter(data['test'], 1, 0, gap=None)
    >>> data
        test  test_rep
    0      0         0
    1      1         1
    2      1         0
    3      0         0
    4     -1        -1
    5     -1        -1
    6      2         2
    7     -1        -1
    8      1         1
    9      0         0
    10     1         0
    11     1         0
    12     1         0
    13     0         0
    14     0         0
    15    -1        -1
    16    -1        -1
    17     0         0
    18     0         0
    19     1         1
    >>> series = pd.Series([-1, 1, -1, 0, 1, 0, 1, 1, -1])
    >>> replace_repeat_iter(series, 1, 0, gap=5)
    0   -1
    1    1
    2   -1
    3    0
    4    1
    5    0
    6    0
    7    0
    8   -1
    '''
    if not keep_last:
        return _replace_repeat_iter(series, val, val0, gap=gap)
    else:
        series_ = series[::-1]
        series_ = _replace_repeat_iter(series_, val, val0, gap=gap)
        return series_[::-1]


def _replace_repeat_iter(series, val, val0, gap=None):
    '''
    TODO
    ----
    改为不在df里面算（df.loc可能会比较慢？）
    '''

    col = series.name
    df = pd.DataFrame({col: series})

    if gap is not None and (gap > df.shape[0] or gap < 1):
        raise ValueError('gap取值范围必须为1到df.shape[0]之间！')
    gap = None if gap == 1 else gap

    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    k = 0
    while k < df.shape[0]:
        if df.loc[df.index[k], col] == val:
            k1 = k + 1

            if gap is None:
                while k1 < df.shape[0] and \
                                    df.loc[df.index[k1], col] in [val, val0]:
                    if df.loc[df.index[k1], col] == val:
                        df.loc[df.index[k1], col] = val0
                    k1 += 1
            else:
                while k1 < min(k+gap, df.shape[0]) and \
                                    df.loc[df.index[k1], col] in [val, val0]:
                    if df.loc[df.index[k1], col] == val:
                        df.loc[df.index[k1], col] = val0
                    k1 += 1
            k =  k1

        else:
            k += 1

    df.index = ori_index

    return df[col]
    
    
def replace_repeat_pd(series, val, val0, keep_last=False):
    '''
    | 替换序列中重复出现的值, 仅保留第一个
    | 
    | 函数功能，参数和意义同 :func:`dramkit.gentools.replace_repeat_iter`
    | 区别在于计算时在pandas.DataFrame里面进行而不是采用迭代方式，同时取消了gap
      参数(即连续出现的val值只保留第一个)
    '''
    if not keep_last:
        return _replace_repeat_pd(series, val, val0)
    else:
        series_ = series[::-1]
        series_ = _replace_repeat_pd(series_, val, val0)
        return series_[::-1]


def _replace_repeat_pd(series, val, val0):
    col = series.name
    df = pd.DataFrame({col: series})
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_ori = col
    col = 'series'
    df.columns = [col]

    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    df['gap1'] = df[col].apply(lambda x: x not in [val, val0]).astype(int)
    df['is_val'] = df[col].apply(lambda x: x == val).astype(int)
    df['val_or_gap'] = df['gap1'] + df['is_val']
    df['pre_gap'] = df[df['val_or_gap'] == 1]['gap1'].shift(1)
    df['pre_gap'] = df['pre_gap'].fillna(method='ffill')
    k = 0
    while k < df.shape[0] and df.loc[df.index[k], 'is_val'] != 1:
        k += 1
    if k < df.shape[0]:
        df.loc[df.index[k], 'pre_gap'] = 1
    df['pre_gap'] = df['pre_gap'].fillna(0).astype(int)
    df['keep1'] = (df['is_val'] + df['pre_gap']).map({0: 0, 1: 0, 2: 1})
    df['to_rplc'] = (df['keep1'] + df['is_val']).map({2: 0, 1: 1, 0: 0})
    df[col] = df[[col, 'to_rplc']].apply(lambda x:
                            val0 if x['to_rplc'] == 1 else x[col], axis=1)

    df.rename(columns={col: col_ori}, inplace=True)
    df.index = ori_index

    return df[col_ori]
    
    
def replace_repeat_func_iter(series, func_val, func_val0,
                             gap=None, keep_last=False):
    '''
    | 替换序列中重复出现的值，功能与 :func:`dramkit.gentools.replace_repeat_iter`
      类似，只不过把val和val0的值由直接指定换成了由指定函数生成

    | ``func_val`` 函数用于判断连续条件，其返回值只能是True或False，
    | ``func_val0`` 函数用于生成替换的新值。
    | 即series中若步长为gap的范围内出现多个满足func_val函数为True的值， 
      则只保留第一条记录，后面的替换为函数func_val0的值。
    | 若gap为None，则将连续出现的满足func_val函数为True的值只保留第一个，其余替换为函数
      func_val0的值(这里连续出现是指不出现除了满足func_val为True和等于func_val0函数值
      之外的其他值)
    
    返回结果为替换之后的series (`pd.Series`)

    Examples
    --------
    >>> data = pd.DataFrame({'y': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1,
    ...                            1, 1, 0, 0, -1, -1, 0, 0, 1]})
    >>> data['y_rep'] = replace_repeat_func_iter(
    ...                 data['y'], lambda x: x < 1, lambda x: 3, gap=None)
    >>> data
        y  y_rep
    0   0      0
    1   1      1
    2   1      1
    3   0      0
    4  -1      3
    5  -1      3
    6   2      2
    7  -1     -1
    8   1      1
    9   0      0
    10  1      1
    11  1      1
    12  1      1
    13  0      0
    14  0      3
    15 -1      3
    16 -1      3
    17  0      3
    18  0      3
    19  1      1
    '''
    if not keep_last:
        return _replace_repeat_func_iter(series, func_val, func_val0, gap=gap)
    else:
        series_ = series[::-1]
        series_ = _replace_repeat_func_iter(series_, func_val, func_val0, gap=gap)
        return series_[::-1]


def _replace_repeat_func_iter(series, func_val, func_val0, gap=None):
    col = series.name
    df = pd.DataFrame({col: series})

    if gap is not None and (gap > df.shape[0] or gap < 1):
        raise ValueError('gap取值范围必须为1到df.shape[0]之间！')
    gap = None if gap == 1 else gap

    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    k = 0
    while k < df.shape[0]:
        if func_val(df.loc[df.index[k], col]):
            k1 = k + 1

            if gap is None:
                while k1 < df.shape[0] and \
                                    (func_val(df.loc[df.index[k1], col]) \
                                     or df.loc[df.index[k1], col] == \
                                         func_val0(df.loc[df.index[k1], col])):
                    if func_val(df.loc[df.index[k1], col]):
                        df.loc[df.index[k1], col] = \
                                              func_val0(df.loc[df.index[k1], col])
                    k1 += 1
            else:
                while k1 < min(k+gap, df.shape[0]) and \
                                (func_val(df.loc[df.index[k1], col]) \
                                 or df.loc[df.index[k1], col] == \
                                     func_val0(df.loc[df.index[k1], col])):
                    if func_val(df.loc[df.index[k1], col]):
                        df.loc[df.index[k1], col] = \
                                              func_val0(df.loc[df.index[k1], col])
                    k1 += 1
            k =  k1

        else:
            k += 1

    df.index = ori_index

    return df[col]
    
    
def replace_repeat_func_pd(series, func_val, func_val0, keep_last=False):
    '''
    替换序列中重复出现的值, 仅保留第一个
    
    | 函数功能，参数和意义同 :func:`dramkit.gentools.replace_repeat_func_iter`
    | 区别在于计算时在pandas.DataFrame里面进行而不是采用迭代方式
    | 同时取消了gap参数(即连续出现的满足func_val为True的值只保留第一个)
    '''
    if not keep_last:
        return _replace_repeat_func_pd(series, func_val, func_val0)
    else:
        series_ = series[::-1]
        series_ = _replace_repeat_func_pd(series_, func_val, func_val0)
        return series_[::-1]


def _replace_repeat_func_pd(series, func_val, func_val0):
    col = series.name
    df = pd.DataFrame({col: series})
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_ori = col
    col = 'series'
    df.columns = [col]

    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    df['gap1'] = df[col].apply(lambda x:
                               not func_val(x) and x != func_val0(x)).astype(int)
    df['is_val'] = df[col].apply(lambda x: func_val(x)).astype(int)
    df['val_or_gap'] = df['gap1'] + df['is_val']
    df['pre_gap'] = df[df['val_or_gap'] == 1]['gap1'].shift(1)
    df['pre_gap'] = df['pre_gap'].fillna(method='ffill')
    k = 0
    while k < df.shape[0] and df.loc[df.index[k], 'is_val'] != 1:
        k += 1
    if k < df.shape[0]:
        df.loc[df.index[k], 'pre_gap'] = 1
    df['pre_gap'] = df['pre_gap'].fillna(0).astype(int)
    df['keep1'] = (df['is_val'] + df['pre_gap']).map({0: 0, 1: 0, 2: 1})
    df['to_rplc'] = (df['keep1'] + df['is_val']).map({2: 0, 1: 1, 0: 0})
    df[col] = df[[col, 'to_rplc']].apply(
              lambda x: func_val0(x[col]) if x['to_rplc'] == 1 else x[col],
              axis=1)

    df.rename(columns={col: col_ori}, inplace=True)
    df.index = ori_index

    return df[col_ori]


def con_count(series, func_cond, via_pd=True):
    '''
    计算series(pd.Series)中连续满足func_cond函数指定的条件的记录数

    Parameters
    ----------
    series : pd.Series
        目标序列
    func_cond : function
        指定条件的函数，func_cond(x)返回结果只能为True或False
    via_pd : bool
        若via_pd为False，则计算时使用循环迭代，否则在pandas.DataFrame里面进行计算


    :returns: `pd.Series` - 返回连续计数结果

    Examples
    --------
    >>> df = pd.DataFrame([0, 0, 1, 1, 0, 0, 1, 1, 1], columns=['series'])
    >>> func_cond = lambda x: True if x == 1 else False
    >>> df['count1'] = con_count(df['series'], func_cond, True)
    >>> df
       series  count1
    0       0       0
    1       0       0
    2       1       1
    3       1       2
    4       0       0
    5       0       0
    6       1       1
    7       1       2
    8       1       3
    >>> df['count0'] = con_count(df['series'], lambda x: x != 1, False)
    >>> df
       series  count1  count0
    0       0       0       1
    1       0       0       2
    2       1       1       0
    3       1       2       0
    4       0       0       1
    5       0       0       2
    6       1       1       0
    7       1       2       0
    8       1       3       0
    '''

    col = 'series'
    series.name = col
    df = pd.DataFrame(series)

    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    if via_pd:
        df['Fok'] = df[col].apply(lambda x: func_cond(x)).astype(int)
        df['count'] = df['Fok'].cumsum()
        df['tmp'] = df[df['Fok'] == 0]['count']
        df['tmp'] = df['tmp'].fillna(method='ffill')
        df['tmp'] = df['tmp'].fillna(0)
        df['count'] = (df['count'] - df['tmp']).astype(int)

        df.index = ori_index

        return df['count']

    else:
        df['count'] = 0
        k = 0
        while k < df.shape[0]:
            if func_cond(df.loc[df.index[k], col]):
                count = 1
                df.loc[df.index[k], 'count'] = count
                k1 = k + 1
                while k1 < df.shape[0] and func_cond(df.loc[df.index[k1], col]):
                    count += 1
                    df.loc[df.index[k1], 'count'] = count
                    k1 += 1
                k = k1
            else:
                k += 1

        df.index = ori_index

        return df['count']


def con_count_ignore(series, func_cond, via_pd=True, func_ignore=None):
    '''
    在 :func:`dramkit.gentools.con_count` 的基础上增加连续性判断条件:

        当series中的值满足func_ignore函数值为True时，不影响连续性判断(func_ignore
        默认为 ``lambda x: isnull(x)``)
    '''
    if func_ignore is None:
        func_ignore = lambda x: isnull(x)
    df = pd.DataFrame({'v': series})
    df['ignore'] = df['v'].apply(lambda x: func_ignore(x)).astype(int)
    df['count'] = con_count(df[df['ignore'] == 0]['v'], func_cond, via_pd=via_pd)
    df['count'] = df['count'].fillna(0)
    df['count'] = df['count'].astype(int)
    return df['count']


def get_preval_func_cond(data, col_val, col_cond, func_cond):
    '''
    | 获取上一个满足指定条件的行中col_val列的值，条件为：
    | 该行中col_cond列的值x满足func_cond(x)为True (func_cond(x)返回结果只能为True或False)
    | 返回结果为 `pd.Series`

    Examples
    --------
    >>> data = pd.DataFrame({'x1': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, 1, 1, 1,
    ...                             0, 0, -1, -1, 0, 0, 1],
    ...                      'x2': [0, 1, 1, 0, -1, -1, 1, -1, 1, 0, 1, 1, 1,
    ...                             0, 0, -1, -1, 0, 0, 1]})
    >>> data['x1_pre'] = get_preval_func_cond(data, 'x1', 'x2', lambda x: x != 1)
    >>> data
        x1  x2  x1_pre
    0    0   0     NaN
    1    1   1     0.0
    2    1   1     0.0
    3    0   0     0.0
    4   -1  -1     0.0
    5   -1  -1    -1.0
    6    2   1    -1.0
    7   -1  -1    -1.0
    8    1   1    -1.0
    9    0   0    -1.0
    10   1   1     0.0
    11   1   1     0.0
    12   1   1     0.0
    13   0   0     0.0
    14   0   0     0.0
    15  -1  -1     0.0
    16  -1  -1    -1.0
    17   0   0    -1.0
    18   0   0     0.0
    19   1   1     0.0
    '''

    df = data[[col_val, col_cond]].copy()
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_val, col_cond = ['col_val', 'col_cond']
    df.columns = [col_val, col_cond]

    df['Fok'] = df[col_cond].apply(lambda x: func_cond(x)).astype(int)
    df['val_pre'] = df[df['Fok'] == 1][col_val]
    df['val_pre'] = df['val_pre'].shift(1).fillna(method='ffill')

    return df['val_pre']


def gap_count(series, func_cond, via_pd=True):
    '''
    计算series (`pd.Series`)中当前行距离上一个满足 ``func_cond`` 函数指定条件记录的行数
    
    func_cond为指定条件的函数，func_cond(x)返回结果只能为True或False，
    若via_pd为False，则使用循环迭代，若via_pd为True，则在pandas.DataFrme内计算
    返回结果为 `pd.Series`

    Examples
    --------
    >>> df = pd.DataFrame([0, 1, 1, 0, 0, 1, 1, 1], columns=['series'])
    >>> func_cond = lambda x: True if x == 1 else False
    >>> df['gap1'] = gap_count(df['series'], func_cond, True)
    >>> df
       series  gap1
    0       0     0
    1       1     0
    2       1     1
    3       0     1
    4       0     2
    5       1     3
    6       1     1
    7       1     1
    >>> df['gap0'] = gap_count(df['series'], lambda x: x != 1, False)
    >>> df
       series  gap1  gap0
    0       0     0     0
    1       1     0     1
    2       1     1     2
    3       0     1     3
    4       0     2     1
    5       1     3     1
    6       1     1     2
    7       1     1     3
    '''

    col = 'series'
    series.name = col
    df = pd.DataFrame(series)

    # 当series.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    if via_pd:
        df['idx'] = range(0, df.shape[0])
        df['idx_pre'] = get_preval_func_cond(df, 'idx', col, func_cond)
        df['gap'] = (df['idx'] - df['idx_pre']).fillna(0).astype(int)

        df.index = ori_index

        return df['gap']

    else:
        df['count'] = con_count(series, lambda x: not func_cond(x), via_pd=via_pd)

        df['gap'] = df['count']
        k0 = 0
        while k0 < df.shape[0] and not func_cond(df.loc[df.index[k0], col]):
            df.loc[df.index[k0], 'gap'] = 0
            k0 += 1

        for k1 in range(k0+1, df.shape[0]):
            if func_cond(df.loc[df.index[k1], col]):
                df.loc[df.index[k1], 'gap'] = \
                                        df.loc[df.index[k1-1], 'count'] + 1

        df.index = ori_index

        return df['gap']


def count_between_gap(data, col_gap, col_count, func_gap, func_count,
                      count_now_gap=False, count_now=True, via_pd=True):
    '''
    计算data (`pandas.DataFrame`)中当前行与上一个满足 ``func_gap`` 函数为True的行之间，
    满足 ``func_count`` 函数为True的记录数

    | 函数func_gap作用于 ``col_gap`` 列，func_count作用于 ``col_count`` 列，
      两者返回值均为True或False
    | ``count_now_gap`` 设置满足func_gap的行是否参与计数，若为False，
      则该行计数为0，若为True，则该行按照上一次计数的最后一次计数处理
    
    .. todo::
        增加count_now_gap的处理方式：
        
        - 该行计数为0
        - 该行按上一次计数的最后一次计数处理
        - 该行按下一次计数的第一次计数处理

    ``count_now`` 设置当当前行满足func_count时，从当前行开始对其计数还是从下一行开始对其计数
    
    .. note::
        注：当前行若满足同时满足func_gap和func_count，对其计数的行不会为下一行
        (即要么不计数，要么在当前行对其计数)

    若via_pd为True，则调用 :func:`count_between_gap_pd` 实现，否则用 :func:`count_between_gap_iter`

    返回结果为 `pd.Series`

    Examples
    --------
    >>> data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, -1, 1,
    ...                                 1, 0, 0, -1, -1, 0, 0, 1],
    ...                      'to_count': [0, 1, 1, 0, -1, -1, 1, -1, 1, 0, 1,
    ...                                   1, 1, 0, 0, -1, -1, 0, 0, 1]})
    >>> data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
    ...                                       lambda x: x == -1, lambda x: x == 1,
    ...                                       count_now_gap=False, count_now=False)
    >>> data
            to_gap  to_count  gap_count
    0        0         0          0
    1        1         1          0
    2        1         1          0
    3        0         0          0
    4       -1        -1          0
    5       -1        -1          0
    6        2         1          0
    7       -1        -1          0
    8        1         1          0
    9        0         0          1
    10      -1         1          0
    11       1         1          0
    12       1         1          1
    13       0         0          2
    14       0         0          2
    15      -1        -1          0
    16      -1        -1          0
    17       0         0          0
    18       0         0          0
    19       1         1          0
    >>> data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, -1, 1,
                                        1, 0, 0, -1, -1, 0, 0, 1, -1, -1],
                             'to_count': [0, 1, 1, 0, -1, -1, 1, -1, 1, 0, 1,
                                          1, 1, 0, 0, -1, 1, 0, 1, 1, 1, 1]})
    >>> data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                              lambda x: x == -1, lambda x: x == 1,
                                              count_now_gap=False, count_now=True)
    >>> data
            to_gap  to_count  gap_count
    0        0         0          0
    1        1         1          0
    2        1         1          0
    3        0         0          0
    4       -1        -1          0
    5       -1        -1          0
    6        2         1          1
    7       -1        -1          0
    8        1         1          1
    9        0         0          1
    10      -1         1          0
    11       1         1          1
    12       1         1          2
    13       0         0          2
    14       0         0          2
    15      -1        -1          0
    16      -1         1          0
    17       0         0          0
    18       0         1          1
    19       1         1          2
    20      -1         1          0
    21      -1         1          0
    >>> data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, -1, 1,
                                        1, 0, 0, -1, -1, 0, 0, 1, -1, -1],
                             'to_count': [0, -1, -1, 0, -1, -1, 1, -1, 1, 0, 1, 1,
                                          1, 0, 0, -1, -1, 0, -1, 1, 1, 1]})
    >>> data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                              lambda x: x == -1, lambda x: x == 1,
                                              count_now_gap=True, count_now=False)
    >>> data
            to_gap  to_count  gap_count
    0        0         0          0
    1        1        -1          0
    2        1        -1          0
    3        0         0          0
    4       -1        -1          0
    5       -1        -1          0
    6        2         1          0
    7       -1        -1          1
    8        1         1          0
    9        0         0          1
    10      -1         1          1
    11       1         1          0
    12       1         1          1
    13       0         0          2
    14       0         0          2
    15      -1        -1          2
    16      -1        -1          0
    17       0         0          0
    18       0        -1          0
    19       1         1          0
    20      -1         1          1
    21      -1         1          0
    >>> data = pd.DataFrame({'to_gap': [0, 1, 1, 0, -1, -1, 2, -1, 1, 0, -1, 1,
                                        1, 0, 0, -1, -1, 0, 0, 1, -1, -1],
                             'to_count': [0, -1, -1, 0, -1, -1, 1, -1, 1, 0, 1, 1,
                                          1, 0, 0, -1, -1, 0, -1, 1, 1, 1]})
    >>> data['gap_count'] = count_between_gap(data, 'to_gap', 'to_count',
                                              lambda x: x == -1, lambda x: x == 1,
                                              count_now_gap=True, count_now=True)
    >>> data
            to_gap  to_count  gap_count
    0        0         0          0
    1        1        -1          0
    2        1        -1          0
    3        0         0          0
    4       -1        -1          0
    5       -1        -1          0
    6        2         1          1
    7       -1        -1          1
    8        1         1          1
    9        0         0          1
    10      -1         1          2
    11       1         1          1
    12       1         1          2
    13       0         0          2
    14       0         0          2
    15      -1        -1          2
    16      -1        -1          0
    17       0         0          0
    18       0        -1          0
    19       1         1          1
    20      -1         1          2
    21      -1         1          1
    '''

    if via_pd:
        return count_between_gap_pd(data, col_gap, col_count, func_gap,
                                    func_count, count_now_gap=count_now_gap,
                                    count_now=count_now)
    else:
        return count_between_gap_iter(data, col_gap, col_count, func_gap,
                                      func_count, count_now_gap=count_now_gap,
                                      count_now=count_now)


def count_between_gap_pd(data, col_gap, col_count, func_gap, func_count,
                         count_now_gap=True, count_now=True):
    '''参数和功能说明见 :func:`dramkit.gentools.count_between_gap` 函数'''

    df = data[[col_gap, col_count]].copy()
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_gap, col_count = ['col_gap', 'col_count']
    df.columns = [col_gap, col_count]

    df['gap0'] = df[col_gap].apply(lambda x: not func_gap(x)).astype(int)
    df['count1'] = df[col_count].apply(lambda x: func_count(x)).astype(int)
    df['gap_count'] = df[df['gap0'] == 1]['count1'].cumsum()
    df['gap_cut'] = df['gap0'].diff().shift(-1)
    df['gap_cut'] = df['gap_cut'].apply(lambda x: 1 if x == -1 else np.nan)
    df['tmp'] = (df['gap_count'] * df['gap_cut']).shift(1)
    df['tmp'] = df['tmp'].fillna(method='ffill')
    df['gap_count'] = df['gap_count'] - df['tmp']

    if count_now_gap:
        df['pre_gap0'] = df['gap0'].shift(1)
        df['tmp'] = df['gap_count'].shift()
        df['tmp'] = df[df['gap0'] == 0]['tmp']

        df['gap_count1'] = df['gap_count'].fillna(0)
        df['gap_count2'] = df['tmp'].fillna(0) + df['count1'] * (1-df['gap0'])
        df['gap_count'] = df['gap_count1'] + df['gap_count2']

    if not count_now:
        df['gap_count'] = df['gap_count'].shift(1)
        if not count_now_gap:
            df['gap_count'] = df['gap0'] * df['gap_count']
        else:
            df['gap_count'] = df['pre_gap0'] * df['gap_count']

    df['gap_count'] = df['gap_count'].fillna(0).astype(int)

    return df['gap_count']


def count_between_gap_iter(data, col_gap, col_count, func_gap, func_count,
                           count_now_gap=True, count_now=True):
    '''参数和功能说明见 :func:`dramkit.gentools.count_between_gap` 函数'''

    df = data[[col_gap, col_count]].copy()
    # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
    col_gap, col_count = ['col_gap', 'col_count']
    df.columns = [col_gap, col_count]

    # 当data.index存在重复值时为避免报错，因此先重置index最后再还原
    ori_index = df.index
    df.index = range(0, df.shape[0])

    df['gap_count'] = 0

    k = 0
    while k < df.shape[0]:
        if func_gap(df.loc[df.index[k], col_gap]):
            k += 1
            gap_count = 0
            while k < df.shape[0] and \
                                  not func_gap(df.loc[df.index[k], col_gap]):
                if func_count(df.loc[df.index[k], col_count]):
                    gap_count += 1
                df.loc[df.index[k], 'gap_count'] = gap_count
                k += 1
        else:
            k += 1

    if count_now_gap:
        k = 1
        while k < df.shape[0]:
            if func_gap(df.loc[df.index[k], col_gap]):
                if not func_gap(df.loc[df.index[k-1], col_gap]):
                    if func_count(df.loc[df.index[k], col_count]):
                        df.loc[df.index[k], 'gap_count'] = \
                                        df.loc[df.index[k-1], 'gap_count'] + 1
                        k += 1
                    else:
                        df.loc[df.index[k], 'gap_count'] = \
                                            df.loc[df.index[k-1], 'gap_count']
                        k += 1
                else:
                    if func_count(df.loc[df.index[k], col_count]):
                        df.loc[df.index[k], 'gap_count'] = 1
                        k += 1
                    else:
                        k += 1
            else:
                k += 1

    if not count_now:
        df['gap_count_pre'] = df['gap_count'].copy()
        if not count_now_gap:
            for k in range(1, df.shape[0]):
                if func_gap(df.loc[df.index[k], col_gap]):
                    df.loc[df.index[k], 'gap_count'] = 0
                else:
                    df.loc[df.index[k], 'gap_count'] = \
                                        df.loc[df.index[k-1], 'gap_count_pre']
        else:
            for k in range(1, df.shape[0]):
                if func_gap(df.loc[df.index[k-1], col_gap]):
                    df.loc[df.index[k], 'gap_count'] = 0
                else:
                    df.loc[df.index[k], 'gap_count'] = \
                                        df.loc[df.index[k-1], 'gap_count_pre']
        df.drop('gap_count_pre', axis=1, inplace=True)

    k0 = 0
    while k0 < df.shape[0] and not func_gap(df.loc[df.index[k0], col_gap]):
        df.loc[df.index[k0], 'gap_count'] = 0
        k0 += 1
    df.loc[df.index[k0], 'gap_count'] = 0

    df.index = ori_index

    return df['gap_count']


def val_gap_cond(data, col_val, col_cond, func_cond, func_val,
                 to_cal_col=None, func_to_cal=None, val_nan=np.nan,
                 contain_1st=False):
    '''
    计算data (`pandas.DataFrame`)中从上一个 ``col_cond`` 列满足 ``func_cond`` 函数的行
    到当前行, ``col_val`` 列记录的 ``func_val`` 函数值

    | func_cond作用于col_cond列，func_cond(x)返回True或False，x为单个值
    | func_val函数作用于col_val列，func_val(x)返回单个值，x为np.array或pd.Series或列表等
    | func_to_cal作用于to_cal_col列，只有当前行func_to_cal值为True时才进行func_val计算，
      否则返回结果中当前行值设置为val_nan
    | contain_1st设置func_val函数计算时是否将上一个满足func_cond的行也纳入计算

    .. todo::
        参考 :func:`dramkit.gentools.count_between_gap` 的设置:

        - 设置col_cond列满足func_cond函数的行，其参与func_val函数的前一次计算还是下一次计算还是不参与计算

    Examples
    --------
    >>> data = pd.DataFrame({'val': [1, 2, 5, 3, 1, 7 ,9],
    ...                      'sig': [1, 1, -1, 1, 1, -1, 1]})
    >>> data['val_pre1'] = val_gap_cond(data, 'val', 'sig',
    ...                    lambda x: x == -1, lambda x: max(x))
    >>> data
       val  sig  val_pre1
    0    1    1       NaN
    1    2    1       NaN
    2    5   -1       NaN
    3    3    1       3.0
    4    1    1       3.0
    5    7   -1       7.0
    6    9    1       9.0
    '''

    if to_cal_col is None and func_to_cal is None:
        df = data[[col_val, col_cond]].copy()
        # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
        col_val, col_cond = ['col_val', 'col_cond']
        df.columns = [col_val, col_cond]
    elif to_cal_col is not None and func_to_cal is not None:
        df = data[[col_val, col_cond, to_cal_col]].copy()
        # 为了避免计算过程中临时产生的列名与原始列名混淆，对列重新命名
        col_val, col_cond, to_cal_col = ['col_val', 'col_cond',
                                                               'to_cal_col']
        df.columns = [col_val, col_cond, to_cal_col]

    df['idx'] = range(0, df.shape[0])
    df['pre_idx'] = get_preval_func_cond(df, 'idx', col_cond, func_cond)

    if to_cal_col is None and func_to_cal is None:
        if not contain_1st:
            df['gap_val'] = df[['pre_idx', 'idx', col_val]].apply(lambda x:
               func_val(df[col_val].iloc[int(x['pre_idx']+1): int(x['idx']+1)]) \
               if not isnull(x['pre_idx']) else val_nan, axis=1)
        else:
            df['gap_val'] = df[['pre_idx', 'idx', col_val]].apply(lambda x:
               func_val(df[col_val].iloc[int(x['pre_idx']): int(x['idx']+1)]) \
               if not isnull(x['pre_idx']) else val_nan, axis=1)
    elif to_cal_col is not None and func_to_cal is not None:
        if not contain_1st:
            df['gap_val'] = df[['pre_idx', 'idx', col_val,
                                                to_cal_col]].apply(lambda x:
              func_val(df[col_val].iloc[int(x['pre_idx']+1): int(x['idx']+1)]) \
              if not isnull(x['pre_idx']) and func_to_cal(x[to_cal_col]) else \
              val_nan, axis=1)
        else:
            df['gap_val'] = df[['pre_idx', 'idx', col_val,
                                                to_cal_col]].apply(lambda x:
              func_val(df[col_val].iloc[int(x['pre_idx']): int(x['idx']+1)]) \
              if not isnull(x['pre_idx']) and func_to_cal(x[to_cal_col]) else \
              val_nan, axis=1)

    return df['gap_val']


def filter_by_func_prenext(l, func_prenext):
    '''
    对 ``l`` (`list`)进行过滤，过滤后返回的 ``lnew`` (`list`)任意前后相邻两个元素满足:

        func_prenext(lnew[i], lnew[i+1]) = True

    过滤过程为：将 ``l`` 的第一个元素作为起点，找到其后第一个满足 ``func_prenext`` 函数
    值为True的元素，再以该元素为起点往后寻找...

    Examples
    --------
    >>> l = [1, 2, 3, 4, 1, 1, 2, 3, 6]
    >>> func_prenext = lambda x, y: (y-x) >= 2
    >>> filter_by_func_prenext(l, func_prenext)
    [1, 3, 6]
    >>> l = [1, 2, 3, 4, 1, 5, 1, 2, 3, 6]
    >>> filter_by_func_prenext(l, func_prenext)
    [1, 3, 5]
    >>> filter_by_func_prenext(l, lambda x, y: y == x+1)
    [1, 2, 3, 4]
    >>> l = [(1, 2), (2, 3), (4, 1), (5, 0)]
    >>> func_prenext = lambda x, y: abs(y[-1]-x[-1]) == 1
    >>> filter_by_func_prenext(l, func_prenext)
    [(1, 2), (2, 3)]
    '''

    if len(l) == 0:
        return l

    lnew = [l[0]]
    idx_pre, idx_post = 0, 1
    while idx_post < len(l):
        vpre = l[idx_pre]
        idx_post = idx_pre + 1

        while idx_post < len(l):
            vpost = l[idx_post]

            if not func_prenext(vpre, vpost):
                idx_post += 1
            else:
                lnew.append(vpost)
                idx_pre = idx_post
                break

    return lnew


def filter_by_func_prenext_series(series, func_prenext,
                                  func_ignore=None, val_nan=np.nan):
    '''
    对series (`pandas.Series`)调用 ``filter_by_func_prenext`` 函数进行过滤，
    其中满足 ``func_ignore`` 函数为True的值不参与过滤，func_ignore函数默认为：
    ``lambda x: isnull(x)``

    series中 **被过滤的值** 在返回结果中用 ``val_nan`` 替换, **不参与过滤** 的值保持不变
    
    See Also
    --------
    :func:`dramkit.gentools.filter_by_func_prenext`

    Examples
    --------
    >>> series = pd.Series([1, 2, 3, 4, 1, 1, 2, 3, 6])
    >>> func_prenext = lambda x, y: (y-x) >= 2
    >>> filter_by_func_prenext_series(series, func_prenext)
    0    1.0
    1    NaN
    2    3.0
    3    NaN
    4    NaN
    5    NaN
    6    NaN
    7    NaN
    8    6.0
    >>> series = pd.Series([1, 2, 0, 3, 0, 4, 0, 1, 0, 0, 1, 2, 3, 6],
    ...                    index=range(14, 0, -1))
    >>> filter_by_func_prenext_series(series, func_prenext, lambda x: x == 0)
    14    1.0
    13    NaN
    12    0.0
    11    3.0
    10    0.0
    9     NaN
    8     0.0
    7     NaN
    6     0.0
    5     0.0
    4     NaN
    3     NaN
    2     NaN
    1     6.0
    '''
    
    if func_ignore is None:
        func_ignore = lambda x: isnull(x)

    l = [[k, series.iloc[k]] for k in range(0, len(series)) \
                                             if not func_ignore(series.iloc[k])]
    lnew = filter_by_func_prenext(l, lambda x, y: func_prenext(x[1], y[1]))

    i_l = [k for k, v in l]
    i_lnew = [k for k, v in lnew]
    idxs_ignore = [_ for _ in i_l if _ not in i_lnew]

    seriesNew = series.copy()
    for k in idxs_ignore:
        seriesNew.iloc[k] = val_nan

    return seriesNew


def merge_df(df_left, df_right, same_keep='left', **kwargs):
    '''
    在 ``pd.merge`` 上改进，相同列名时自动去除重复的

    Parameters
    ----------
    df_left : pandas.DataFrame
        待merge左表
    df_right : pandas.DataFrame
        待merge右表
    same_keep : str
        可选'left', 'right'，设置相同列保留左边df还是右边df
    **kwargs :
        pd.merge接受的其他参数


    :returns: `pandas.DataFrame` - 返回merge之后的数据表
    '''
    same_cols = [x for x in df_left.columns if x in df_right.columns]
    if len(same_cols) > 0:
        if 'on' in kwargs:
            if isinstance(kwargs['on'], list):
                same_cols = [x for x in same_cols if x not in kwargs['on']]
            elif isinstance(kwargs['on'], str):
                same_cols = [x for x in same_cols if x != kwargs['on']]
            else:
                raise ValueError('on参数只接受list或str！')
        if same_keep == 'left':
            df_right = df_right.drop(same_cols, axis=1)
        elif same_keep == 'right':
            df_left = df_left.drop(same_cols, axis=1)
        else:
            raise ValueError('same_keep参数只接受`left`或`right`！')
    return pd.merge(df_left, df_right, **kwargs)


def cut_df_by_con_val(df, by_col, func_eq=None):
    '''
    根据 `by_col` 列的值，将 `df (pandas.DataFrame)` 切分为多个子集列表，返回 `list`
    
    切分依据：``func_eq`` 函数作用于 ``by_col`` 列，函数值连续相等的记录被划分到一个子集中

    Examples
    --------
    >>> df = pd.DataFrame({'val': range(0,10),
    ...                    'by_col': ['a']*3+['b']*2+['c']*1+['a']*3+['d']*1})
    >>> df.index = ['z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q']
    >>> cut_df_by_con_val(df, 'by_col')
    [   val by_col
     z    0      a
     y    1      a
     x    2      a,
        val by_col
     w    3      b
     v    4      b,
        val by_col
     u    5      c,
        val by_col
     t    6      a
     s    7      a
     r    8      a,
        val by_col
     q    9      d]
    '''

    if isnull(func_eq):
        func_eq = lambda x: x
    df = df.copy()
    df['val_func_eq'] = df[by_col].apply(func_eq)
    by_col = 'val_func_eq'

    sub_dfs= []
    k = 0
    while k < df.shape[0]:
        k1 = k + 1
        while k1 < df.shape[0] and df[by_col].iloc[k1] == df[by_col].iloc[k]:
            k1 += 1
        sub_dfs.append(df.iloc[k:k1, :].drop(by_col, axis=1))
        k = k1

    return sub_dfs


def get_con_start_end(series, func_con):
    '''
    找出series (`pandas.Series`)中值连续满足 ``func_con`` 函数值为True的分段起止位置，
    返回起止位置对列表

    Examples
    --------
    >>> series = pd.Series([0, 1, 1, 0, 1, 1, 0, -1, -1, 0, 0, -1, 1, 1, 1, 1, 0, -1])
    >>> start_ends = get_con_start_end(series, lambda x: x == -1)
    >>> start_ends
    [[7, 8], [11, 11], [17, 17]]
    >>> start_ends = get_con_start_end(series, lambda x: x == 1)
    >>> start_ends
    [[1, 2], [4, 5], [12, 15]]
    '''

    start_ends = []
    # df['start'] = 0
    # df['end'] = 0
    start = 0
    N = len(series)
    while start < N:
        if func_con(series.iloc[start]):
            end = start
            while end < N and func_con(series.iloc[end]):
                end += 1
            start_ends.append([start, end-1])
            # df.loc[df.index[start], 'start'] = 1
            # df.loc[df.index[end-1], 'end'] = 1
            start = end + 1
        else:
            start += 1

    return start_ends


def cut_range_to_subs(n, gap):
    '''
    将 ``range(0, n)`` 切分成连续相接的子集:
    ``[range(0, gap), range(gap, 2*gap), ...]``
    '''
    n_ = n // gap
    mod = n % gap
    if mod != 0:
        return [(k*gap, (k+1)*gap) for k in range(0, n_)] + [(gap * n_, n)]
    else:
        return [(k*gap, (k+1)*gap) for k in range(0, n_)]


def check_l_allin_l0(l, l0):
    '''
    判断 ``l (list)`` 中的值是否都是 ``l0 (list)`` 中的元素, 返回True或False

    Examples
    --------
    >>> l = [1, 2, 3, -1, 0]
    >>> l0 = [0, 1, -1]
    >>> check_l_allin_l0(l, l0)
    False
    >>> l = [1, 1, 0, -1, -1, 0, 0]
    >>> l0 = [0, 1, -1]
    >>> check_l_in_l0(l, l0)
    True
    '''
    l_ = set(l)
    l0_ = set(l0)
    return len(l_-l0_) == 0


def check_exist_data(df, x_list, cols=None):
    '''
    依据指定的 ``cols`` 列检查 ``df (pandas.DataFrame)`` 中是否已经存在 ``x_list (list)`` 中的记录，
    返回list，每个元素值为True或False

    Examples
    --------
    >>> df = pd.DataFrame([['1', 2, 3.1, ], ['3', 4, 5.1], ['5', 6, 7.1]],
    ...                   columns=['a', 'b', 'c'])
    >>> x_list, cols = [[3, 4], ['3', 4]], ['a', 'b']
    >>> check_exist_data(df, x_list, cols=cols)
    [False, True]
    >>> check_exist_data(df, [['1', 3.1], ['3', 5.1]], ['a', 'c'])
    [True, True]
    '''

    if not isnull(cols):
        df_ = df.reindex(columns=cols)
    else:
        df_ = df.copy()
    data = df_.to_dict('split')['data']
    return [x in data for x in x_list]


def isnull(x):
    '''判断x是否为无效值(None, nan, x != x)，若是无效值，返回True，否则返回False'''
    if x is None:
        return True
    if x is np.nan:
        return True
    try:
        if x != x:
            return True
    except:
        pass
    return False


def x_div_y(x, y, v_x0=None, v_y0=0, v_xy0=1):
    '''
    x除以y

    - v_xy0为当x和y同时为0时的返回值
    - v_y0为当y等于0时的返回值
    - v_x0为当x等于0时的返回值
    '''
    if x == 0 and y == 0:
        return v_xy0
    if x != 0 and y == 0:
        return v_y0
    if x == 0 and y != 0:
        return 0 if v_x0 is None else v_x0
    return x / y


def power(a, b, return_real=True):
    '''计算a的b次方，return_real设置是否只返回实属部分'''
    c = a ** b
    if isnull(c):
        c = complex(a) ** complex(b)
    if return_real:
        c = c.real
    return c


def log(x, bottom=None):
    '''计算对数, bottom指定底'''
    assert isinstance(bottom, (int, float))
    if isnull(bottom):
        return np.log(x)
    return (np.log(x)) / (np.log(bottom)) # h换底公式


def cal_pct(v0, v1, vv00=1, vv10=-1):
    '''
    计算从v0到v1的百分比变化

    - vv00为当v0的值为0且v1为正时的返回值，v1为负时取负号
    - vv10为当v1的值为0且v0为正时的返回值，v0为负时取负号
    '''
    if isnull(v0) or isnull(v1):
        return np.nan
    if v0 == 0:
        if v1 == 0:
            return 0
        elif v1 > 0:
            return vv00
        elif v1 < 0:
            return -vv00
    elif v1 == 0:
        if v0 > 0:
            return vv10
        elif v0 < 0:
            return -vv10
    elif v0 > 0 and v1 > 0:
        return v1 / v0 - 1
    elif v0 < 0 and v1 < 0:
        return -(v1 / v0 - 1)
    elif v0 > 0 and v1 < 0:
        return v1 / v0 - 1
    elif v0 < 0 and v1 > 0:
        return -(v1 / v0 - 1)


def min_com_multer(l):
    '''求一列数 `l (list)` 的最小公倍数，支持负数和浮点数'''
    l_max = max(l)
    mcm = l_max
    while any([mcm % x != 0 for x in l]):
        mcm += l_max
    return mcm


def max_com_divisor(l):
    '''
    求一列数 `l (list)` 的最大公约数，只支持正整数

    .. note::
        只支持正整数
    '''

    def _isint(x):
        '''判断x是否为整数'''
        tmp = str(x).split('.')
        if len(tmp) == 1 or all([x == '0' for x in tmp[1]]):
            return True
        return False

    if any([x < 1 or not _isint(x) for x in l]):
        raise ValueError('只支持正整数！')

    l_min = min(l)
    mcd = l_min
    while any([x % mcd != 0 for x in l]):
        mcd -= 1

    return mcd


def mcd2_tad(a, b):
    '''
    辗转相除法求a和b的最大公约数，a、b为正数
    
    .. note::
        - a, b应为正数
        - a, b为小数时由于精度问题会不正确
    '''
    if a < b:
        a, b = b, a # a存放较大值，b存放较小值
    if a % b == 0:
        return b
    else:
        return mcd2_tad(b, a % b)


def max_com_divisor_tad(l):
    '''
    用辗转相除法求一列数 `l (list)` 的最大公约数, `l` 元素均为正数

    .. note::
        - l元素均为正数
        - l元素为小数时由于精度问题会不正确

    References
    ----------
    https://blog.csdn.net/weixin_45069761/article/details/107954905
    '''
    
    # g = l[0]
    # for i in range(1, len(l)):
    #     g = mcd2_tad(g, l[i])
    # return g

    return reduce(lambda x, y: mcd2_tad(x, y), l)


def get_first_appear_index(series, values, ascending=False,
                           return_iloc=False):
    '''获取values中的值在series中第一次出现时的索引'''
    raise NotImplementedError


def get_appear_order(series, ascending=True):
    '''
    标注series (`pandas.Series` , 离散值)中重复元素是第几次出现，
    
    返回为 `pandas.Series`，ascending设置返回结果是否按出现次序升序排列

    Examples
    --------
    >>> df = pd.DataFrame({'v': ['A', 'B', 'A', 'A', 'C', 'C']})
    >>> df.index = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> df['nth'] = get_appear_order(df['v'], ascending=False)
    >>> df
       v  nth
    a  A    3
    b  B    1
    c  A    2
    d  A    1
    e  C    2
    f  C    1
    '''
    df = pd.DataFrame({'v': series})
    # df['Iidx'] = range(0, df.shape[0])
    # df['nth_appear'] = df.groupby('v')['Iidx'].rank(ascending=ascending)
    # df['nth_appear'] = df['nth_appear'].astype(int)
    df['nth_appear'] = df.groupby('v').cumcount(ascending=ascending)+1
    return df['nth_appear']


def label_rep_index_str(df):
    '''
    `df (pandas.DataFrame)` 中的index若有重复，对重复的index进行后缀编号，返回新的 `pandas.DataFrame`

    .. note::
        若存在重复的index，则添加后缀编号之后返回的df，其index为str类型

    Examples
    --------
    >>> df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> label_rep_index_str(df)
        0
    0   1
    1   2
    2   3
    3   4
    4   5
    5   6
    6   7
    7   8
    8   9
    9  10
    >>> df.index = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
    >>> label_rep_index_str(df)
          0
    0     1
    0_2   2
    0_3   3
    1     4
    1_2   5
    2     6
    2_2   7
    2_3   8
    2_4   9
    3    10
    '''
    if df.index.duplicated().sum() == 0:
        return df
    else:
        df = df.copy()
        idx_name = df.index.name
        df['_idx_idx_idx_'] = df.index
        df['_idx_idx_idx_'] = df['_idx_idx_idx_'].astype(str)
        df['_tmp_tmp_'] = get_appear_order(df['_idx_idx_idx_'])
        df['_idx_idx_idx_'] = df[['_idx_idx_idx_', '_tmp_tmp_']].apply(
            lambda x: x['_idx_idx_idx_'] if x['_tmp_tmp_'] == 1 else \
                      '{}_{}'.format(x['_idx_idx_idx_'], x['_tmp_tmp_']),
                      axis=1)
        df.drop('_tmp_tmp_', axis=1, inplace=True)
        df.set_index('_idx_idx_idx_', inplace=True)
        df.index.name = idx_name
        return df


def drop_index_duplicates(df, keep='first'):
    '''删除 ``df (pandas.DataFrame)`` 中index重复的记录'''
    return df[~df.index.duplicated(keep=keep)]


def count_values(df, cols=None):
    '''计算df列中cols指定列的值出现次数'''
    if cols is None:
        cols = list(df.columns)
    elif isinstance(cols, str):
        cols = [cols]
    assert isinstance(cols, (list, tuple))
    tmp = df.reindex(columns=cols)
    tmp['count'] = 1
    tmp = tmp.groupby(cols, as_index=False)['count'].count()
    df_ = pd.merge(df, tmp, how='left', on=cols)
    df_.index = df.index
    return df_['count']


def count_index(df):
    '''计算df的每个index出现的次数'''
    df_index = pd.DataFrame({'index_': df.index})
    df_index['count'] = count_values(df_index, 'index_')
    df_index.index = df_index['index_']
    return df_index['count']


def group_shift():
    '''分组shift，待实现'''
    raise NotImplementedError
    
    
def group_fillna(df, col_fill, cols_groupby, return_all=False,
                 **kwargs_fillna):
    '''
    分组缺失值填充
    '''
    if isinstance(cols_groupby, str):
        cols_groupby = [cols_groupby]
    series_fill = df.groupby(cols_groupby)[col_fill].fillna(**kwargs_fillna)
    series_fill.index = df.index
    return series_fill
    

def group_rank(df, col_rank, cols_groupby,
               return_all=False, **kwargs_rank):
    '''
    分组排序
    
    对df(`pandas.DataFrame`)中 ``cols_rank`` 指定列按 ``cols_groupby`` 指定列分组排序
    
    TODO
    ----
    col_rank可为列表，此时范围dataframe，return_all设置只返回指定列结果或者返回全部dataframe
    
    Parameters
    ----------
    df : pandas.DataFrame
        待排序数据表
    col_rank : str
        需要排序的列
    cols_groupby : str, list
        分组依据列
    **kwargs_rank :
        pandas中rank函数接受的参数
    
    Returns
    -------
    series_rank : pandas.Series
        排序结果
    '''
    assert isinstance(col_rank, str), '`cols_rank`必须为str'
    assert isinstance(cols_groupby, (str, list)), '`cols_groupby`必须为str或list'
    if isinstance(cols_groupby, str):
        cols_groupby = [cols_groupby]
    series_rank = df.groupby(cols_groupby)[col_rank].rank(**kwargs_rank)
    series_rank.index = df.index
    return series_rank


def bootstrapping():
    '''
    bootstraping, 待实现
    '''
    raise NotImplementedError


def groupby_rolling_func(data, cols_groupby, cols_val, func, keep_index=True,
                         kwargs_rolling={}, kwargs_func={}):
    '''
    data按照cols_groupby分组，然后在cols_val列上rolling调用func，    
    func的参数通过kwargs_func设定，
    若cols_val为str，则返回pd.Series；若为list，则返回pd.DataFrame
    若keep_index为True，则返回结果中的index与data一样，否则返回结果中的index由
    cols_groupby设定
    
    待实现
    '''
    # if isinstance(cols_groupby, str):
    #     cols_groupby = [cols_groupby]
    # cols = [cols_val] if isinstance(cols_val, str) else cols_val
    # df = data.reindex(columns=cols_groupby+cols)
    raise NotImplementedError


def link_lists(lists):
    '''
    | 将多个列表连接成一个列表
    | 注：lists为列表，其每个元素也为一个列表
    
    Examples
    --------
    >>> a = [1, 2, 3]
    >>> b = [4, 5, 6]
    >>> c = ['a', 'b']
    >>> d = [a, b]
    >>> link_lists([a, b, c, d])
    [1, 2, 3, 4, 5, 6, 'a', 'b', [1, 2, 3], [4, 5, 6]]
    '''
    assert isinstance(lists, list)
    assert all([isinstance(x, list) for x in lists])
    newlist = []
    for item in lists:
        newlist.extend(item)
    return newlist


def get_num_decimal(x, ignore_tail0=True):
    '''
    | 获取浮点数x的小数位数
    | ignore_tail0设置是否忽略小数点尾部无效的0
    '''
    try:
        float(x)
    except:
        raise ValueError('输入不是有效浮点数，请检查：{}！'.format(x))
    xstr = str(x)
    xsplit = xstr.split('.')
    if len(xsplit) == 1:
        return 0
    if len(xsplit) > 2:
        raise ValueError('输入出错，请检查：{}！'.format(xstr))
    decimal = xsplit[-1]
    if ignore_tail0:
        while decimal[-1] == '0':
            decimal = decimal[:-1]
    return len(decimal)


def sort_dict(d, by='key', reverse=False):
    '''
    对字典排序，by设置依据'key'还是'value'排，reverse同sorted函数参数
    '''
    assert by in ['key', 'value']
    if by == 'key':
        d_ = sorted(d.items(), key=lambda kv: (kv[0], kv[1]), reverse=reverse)
    else:
        d_ = sorted(d.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)
    return dict(d_)


if __name__ == '__main__':
    from dramkit import load_csv, plot_series
    from finfactory.fintools.fintools import cci

    # 50ETF日线行情------------------------------------------------------------
    fpath = './_test/510050.SH_daily_qfq.csv'
    data = load_csv(fpath)
    data.set_index('date', drop=False, inplace=True)

    data['cci'] = cci(data)
    data['cci_100'] = data['cci'].apply(lambda x: 1 if x > 100 else \
                                                    (-1 if x < -100 else 0))

    plot_series(data.iloc[-200:, :], {'close': ('.-k', False)},
                cols_styl_low_left={'cci': ('.-c', False)},
                cols_to_label_info={'cci':
                                [['cci_100', (-1, 1), ('r^', 'bv'), False]]},
                xparls_info={'cci': [(100, 'r', '-', 1.3),
                                     (-100, 'r', '-', 1.3)]},
                figsize=(8, 7), grids=True)

    start_ends_1 = get_con_start_end(data['cci_100'], lambda x: x == -1)
    start_ends1 = get_con_start_end(data['cci_100'], lambda x: x == 1)
    data['cci_100_'] = 0
    for start, end in start_ends_1:
        if end+1 < data.shape[0]:
            data.loc[data.index[end+1], 'cci_100_'] = -1
    for start, end in start_ends1:
        if end+1 < data.shape[0]:
            data.loc[data.index[end+1], 'cci_100_'] = 1

    plot_series(data.iloc[-200:, :], {'close': ('.-k', False)},
                cols_styl_low_left={'cci': ('.-c', False)},
                cols_to_label_info={'cci':
                                [['cci_100_', (-1, 1), ('r^', 'bv'), False]],
                                    'close':
                                [['cci_100_', (-1, 1), ('r^', 'bv'), False]]},
                xparls_info={'cci': [(100, 'r', '-', 1.3),
                                     (-100, 'r', '-', 1.3)]},
                figsize=(8, 7), grids=True)
