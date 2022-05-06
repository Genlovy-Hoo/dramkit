# -*- coding: utf-8 -*-

import collections


if __name__ == '__main__':

    def add(a: int, b: int) -> int:
        return a + b


    class Person(object):
        def __init__(self, name, gender, age, weight):
            self.name = name
            self.gender = gender
            self.age = age
            self.weight = weight
    Person_ = collections.namedtuple('Person', 'name gender age weight')
    def get_person_info():
        return Person_('jjp', 'MALE', 30, 130)
    person = get_person_info()


    Point = collections.namedtuple('Point', ['x', 'y'])
    p = Point(11, y=22) # 可以使用关键字参数和位置参数初始化namedtuple
    print(p[0] + p[1]) # 可以使用索引去获取namedtuple里面的元素
    x, y = p # 可以将namedtuple拆包
    print(x, y)
    print(p.x + p.y) # 使用对应的字段名字也可以获取namedtuple里面的元素                       # 使用类似name=value的样式增加了数据的可读性
    print(Point(x=11, y=22))


    def d(a=[]):
        a.append(100)
        return 2
    print(d.__defaults__)
    d()
    print(d.__defaults__)
    d()
    print(d.__defaults__)


    # http://c.biancheng.net/view/2270.html
    # funA 作为装饰器函数
    def funA(fn):
        print('funcA')
        fn() # 执行传入的fn参数
        print('---')
        return 'return of funcA'
    @funA
    def funB():
        print('return of funcB')
    print(funB)


    # https://www.cnblogs.com/slysky/p/9777424.html
    def w1(func):
        def inner():
            print('------')
            func()
        return inner
    @w1
    def f1():
        print('f1 called')
    @w1
    def f2():
        print('f2 called')
    f1()
    f2()

    def w1(fun):
        print('...装饰器开始装饰...')
        def inner():
            print('...---...')
            fun()
        return inner
    @w1
    def test():
        print('test')
    test()


    print('\n')
    # https://blog.csdn.net/yuyexiaohan/article/details/82860807
    from functools import wraps
    def my_decorator(func):
    	def wrapper(*args, **kwargs):
    		'''decorator'''
    		print('Decorated function...')
    		return func(*args, **kwargs)
    	return wrapper
    @my_decorator
    def test():
    	"""Testword"""
    	print('Test function')
    print(test.__name__)
    print(test.__doc__)
    test()
    print('\n')
    def my_decorator1(func):
    	@wraps(func)
    	def wrapper(*args, **kwargs):
    		'''decorator'''
    		print('Decorated function...')
    		return func(*args, **kwargs)
    	return wrapper
    @my_decorator1
    def test1():
    	"""Testword"""
    	print('Test function')
    print(test1.__name__)
    print(test1.__doc__)
    test1()
