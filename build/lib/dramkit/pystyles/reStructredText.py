# -*- coding: utf-8 -*-
 
class ReStructuredTextStyle:
    '''reStructuredText风格
    用 ``冒号`` 分隔，
    PyCharm默认风格
    :arg augend: 被加数
    :type augend: int
    '''
 
    def __init__(self, augend, name='ReStructuredTextStyle'):
        '''初始化'''
        self.augend = augend
        self.name = name
 
    def add(self, addend):
        '''加法
        reStructuredText风格的函数，
        类型主要有param、type、return、rtype、exception
        :param addend: 被加数
        :type addend: int
        :returns: 加法结果
        :rtype: :obj:`int` or :obj:`str`
        :exception TypeError: Addition by str
        >>> reStructredText = ReStructuredTextStyle(augend=10)
        >>> reStructredText.add(10)
        20
        '''
        try:
            if isinstance(addend, str):
                raise TypeError('Addition by str')
            else:
                return self.augend + addend
        except TypeError as e:
            return e