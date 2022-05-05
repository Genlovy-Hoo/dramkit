# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    '''sigmoid激活函数'''
    return 1.0 / (1 + np.exp(-x))


def softmax(x, max_=True):
    '''
    | softmax函数
    | x为一维np.array或list或pd.Series
    | 若max_为True，则计算时减去x的最大值以防止取指数之后数据溢出
    | 参考：https://www.cnblogs.com/ysugyl/p/12922598.html
    '''
    x = np.array(x)
    if max_:
        x = x - x.max()
    x_exp = np.exp(x)
    return x_exp / x_exp.sum()


def softmax2d(x2d, axis=0, max_=True):
    '''
    | softmax函数
    | x是二维的np.array或list或dataframe
    | 若max_为True，则计算时减去x的最大值以防止取指数之后数据溢出
    | 参考：https://www.cnblogs.com/ysugyl/p/12922598.html
    '''
    X = np.array(x2d)
    assert(len(X.shape) == 2)
    if max_:
        Xmax = X.max(axis=axis)
        if axis == 0:
            X = X - Xmax
        elif axis == 1:
            X = X - Xmax.reshape(-1, 1)
    X_exp = np.exp(X)
    X_soft = X_exp / X_exp.sum(axis=axis, keepdims=True)
    return X_soft


def softplus(x):
    '''softplus激活函数 '''
    return np.log(1 + np.exp(x))


def tanh(x):
    '''tanh激活函数'''
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


if __name__ == '__main__':
    a = [[1, 2, 3], [-1, -2, -3]]
    b = [[1, 2, 3]]
    c = [1, 2, 3]
    
    print('c')
    print(softmax(c))
    print('c, False')
    print(softmax(c, False))
    
    print('a')
    print(softmax2d(a))
    print('a, axis=1')
    print(softmax2d(a, axis=1))
    print('a, False')
    print(softmax2d(a, max_=False))
    print('b')
    print(softmax2d(b))
    print('b, axis=1')
    print(softmax2d(b, axis=1))
    
    try:
        print(softmax2d(c))
    except:
        # from dramkit import simple_logger
        # from dramkit import logger_show
        # logger_show('Error Info:', simple_logger(), 'err')
        import traceback
        err_info = traceback.format_exc()
        print(err_info)
