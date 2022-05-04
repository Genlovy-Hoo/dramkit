# -*- coding: utf-8 -*-

import math
import numpy as np
from functools import reduce
from numpy.random import uniform


def prod(it):
    '''it中的元素连乘'''
    return reduce(lambda x, y: x * y, it)


class TestFuncs(object):
    '''测试函数集，输入x须为np.array'''

    @staticmethod
    def f1(x):
        '''平方和'''
        return np.sum(x**2)

    @staticmethod
    def f2(x):
        '''绝对值之和加上连乘'''
        return sum(abs(x)) + prod(abs(x))

    @staticmethod
    def f3(x):
        '''x[0]^2 + (x[0]+x[1])^2 + (x[0]+x[1]+x[2])^2 + ...'''
        return sum([sum(x[:k]) ** 2 for k in range(len(x)+1)])

    @staticmethod
    def f4(x):
        '''最小绝对值'''
        return min(abs(x))

    @staticmethod
    def f5(x):
        '''最大绝对值'''
        return max(abs(x))

    @staticmethod
    def f6(x):
        d = len(x)
        part1 = 100 * (x[1:d] - (x[0:d-1] ** 2)) ** 2
        part2 = (x[0:d-1] - 1) ** 2
        return np.sum(part1 + part2)

    @staticmethod
    def f7(x):
        return np.sum(abs((x + 0.5)) ** 2)

    @staticmethod
    def f8(x):
       return sum([(k+1) * (x[k] ** 4) for k in range(len(x))]) + uniform(0, 1)

    @staticmethod
    def f9(x):
        return sum(-x*(np.sin(np.sqrt(abs(x)))))

    @staticmethod
    def f10(x):
        return np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * len(x)

    @staticmethod
    def f11(x):
        d = len(x)
        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
        part2 = np.exp(np.sum(np.cos(2 * math.pi * x)) / d) + 20
        o = part1 - part2 + np.exp(1)
        return o

    @staticmethod
    def f12(x):
        return (x[0] - 1) ** 2 + (x[1] + 1) ** 2 + x[2] ** 2 + x[3] ** 2
    
    @staticmethod
    def schwefel(x):
        return sum([sum(x[:k]) ** 2 for k in range(len(x)+1)])

    @staticmethod
    def griewank(x):
        p1 = sum([y**2 for y in x])
        p2 = [np.cos(x[k-1]/math.sqrt(k)) for k in range(1, len(x)+1)]
        p2_ = 1
        for y in p2:
            p2_ *= y
        return p1/4000 - p2_ + 1

    @staticmethod
    def rastrigin(x):
        tmp = [y**2 - 10*np.cos(2*np.pi*y) + 10 for y in x]
        return sum(tmp)
    
    @staticmethod
    def sphere(x):
        return np.sum(x**2)

    @staticmethod
    def rosenbrock(x):
        dim = len(x)
        tmp = [100 * (x[k+1]-x[k]**2) ** 2 + (x[k]-1) ** 2\
                                                   for k in range(0, dim-1)]
        return np.sum(tmp)
    
    @staticmethod
    def ackley(x):
        d = len(x)
        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
        part2 = np.exp(np.sum(np.cos(2 * math.pi * x)) / d)
        o = part1 - part2 + 20 + np.exp(1)
        return o

    @staticmethod
    def ackley2(x):
        x = x - np.pi
        d = len(x)
        part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
        part2 = np.exp(np.sum(np.cos(2 * math.pi * x)) / d)
        o = part1 - part2 + 20 + np.exp(1)
        return o
