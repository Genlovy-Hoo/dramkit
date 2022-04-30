# -*- coding: utf-8 -*-

'''
程序风格参考模板

dramkit(本程序)的注释文本格式主要借鉴Numpy注释风格 [1]_ ，遵循
re-structured text (reST)语法 [2]_ 并参考Google代码风格 [3]_ [4]_。
本程序的说明文档由Sphinx自动生成 [5]_ [6]_ [7]_ [r1]_ [13]_ 。

References
----------
.. [1] Numpy注释风格指南: https://numpydoc.readthedocs.io/en/latest/format.html
.. [2] https://docutils.sourceforge.io/docs/user/rst/quickref.html
.. [3] Google注释风格指南: https://google.github.io/styleguide/pyguide.html
.. [4] https://zh-google-styleguide.readthedocs.io/en/latest/contents/
.. [5] https://blog.csdn.net/lixiaomei0623/article/details/120530642
.. [6] Sphinx语法: https://mathpretty.com/13551.html
.. [7] https://zhuanlan.zhihu.com/p/264647009
.. [r1] https://www.sphinx-doc.org/en/master/
.. [13] https://www.osgeo.cn/sphinx/index.html 引文编号貌似必须用数字才会正常显示？
.. [8] https://blog.csdn.net/dengdi8115/article/details/102077973
.. [9] reStructuredText在线工具: http://rst.ninjs.org/
.. [10] markdown转reST: https://cloudconvert.com/md-to-rst
.. [11] LightGBM文档_

    .. _LightGBM文档: https://lightgbm.readthedocs.io/en/latest/index.html
.. [12] Niu, Mingfei, Hu, et al. A novel hybrid decomposition-ensemble model
       based on VMD and HGWO for container throughput forecasting[J].
       *Applied Mathematical Modelling* , 2018.
'''


def dramkit_funcsytle(a, b='b', *args, **kwargs):
    '''
    本程序中的function代码风格示例

    函数名由 ``小写单词`` 和 ``_`` 拼接而成，这里写函数详细描述，描述可能比较
    长长长长长长长长长长长长，需要换行

    参见 :class:`dramkit.pystyles.dramkit_style.DramkitClassStyle`

    .. todo::
        待完成描述

        - todo1
        - todo2

    Parameters
    ----------
    a : str
        description of argument *a斜体* : string to print

        .. versionadded:: 版本号0.0.5
            xxx版本新增参数

        .. deprecated:: 版本号0.0.5
            参数 `a` 将在0.0.6版本中移除，替代为参数 `aa`

        .. versionchanged:: 版本号0.0.6
            ``a`` 将修改为 ``int``

        .. note:: 可以用note标注一些注意事项

            这里是注意事项，用无序列表列出

            - 注意1111111111111111111111111111111111111111111111111
              111111换行
            - 注意2
        .. hint:: 可以用hint标注一些提示
        .. important:: 用important描述重要内容
        .. tip:: 用tip提示一些小技巧

            这里是小技巧内容，用无序列表列出

            * 技巧1
            * 技巧2
    b: str, default 'b'
        需要打印的字符串，**b加粗** 参数描述可能很长长长长长长长长长长长长长长
        长长长长长一行放不下，需要跨行

        .. warning:: This is a warning admonition.
        .. caution:: This is a caution admonition.

            .. deprecated:: 版本号0.0.5
                参数 `a` 将在0.0.6版本中移除，替代为参数 `aa`
        .. attention:: This is a attention admonition.
    *args :
        不定参数
    **kwargs :
        关键字参数

        .. error:: This is a error admonition.
        .. danger:: This is a danger admonition.

    Returns
    -------
    a : str
        return a
    b : str
        返回b的值，这里的描述可能也很长长长长长长长长长长长长长长长长长长长长长
        长长长长长一行放不下，需要跨行
    str
        value of (a+b)

    Raises
    ------
    ValueError
        if a is not str

    Examples
    --------
    >>> dramkit_docsytle('1', '2', 3, 4, c=5)
    1
    2
    (3, 4)
    {'c': 5}

    See Also
    --------
    :class:`dramkit.pystyles.dramkit_style.DramkitClassStyle`: 本程序中的class代码风格示例


    .. seealso::

        Module :py:mod:`dramkit.pystyles.dramkit_style` 编码风格模块

    Notes
    -----
    Sphinx也支持生成数学公式，比如FFT是离散傅里叶变换的一种快速实现算法，
    其公式为:

    .. math:: X(e^{j\omega}) = x(n)e^{-j\omega n}

    行内公式写法: :math:`a^2 + b^2 = c^2`

    公式块:

    .. math::

        (a + b)^2 = a^2 + 2ab + b^2

        (a - b)^2 = a^2 - 2ab + b^2

    公式对齐:

    .. math::
        (a + b)^2  &=  (a + b)(a + b) \\
                   \\\&=  a^2 + 2ab + b^2

    .. math::

        y_0 &= x_0 \\
        y_t \\\&= (1 - \\alpha) y_{t-1} + \\alpha x_t

    公式编号:

    Euler's identity, equation :eq:`euler`, was elected one of the most
    beautiful mathematical formulas.

    .. math::
        e^{i\pi} + 1 = 0
        :label: euler

    References
    ----------
    Sphinxg数学公式: http://doc.yonyoucloud.com/doc/zh-sphinx-doc/ext/math.html

    我的论文 [i]_

    .. [i] Niu, Mingfei, Hu, et al. A novel hybrid decomposition-ensemble model
       based on VMD and HGWO for container throughput forecasting[J].
       Applied Mathematical Modelling, 2018.

    .. code-block:: python
       :linenos:
    
       def hello():
           print('这里是Python代码块，linenos用于设置生成行号')

    Note
    ----
    请注意Notes和Note的区别

    为什么生成的文档中引用位置 [i]_ 处有阴影，怎么去除？

    公式编号位置是乱的？

    Sphinx如何安装扩展库？
    '''

    if not isinstance(a, str):
        raise ValueError('`a` must be str!')

    print(a)
    print(b)
    print(args)
    print(kwargs)

    return a, b, a+b


class DramkitClassStyle(object):
    '''
    本程序中的class代码风格示例

    类名由 ``首字母大写的单词`` 拼接而成，这里写函数详细描述，描述可能比较
    长长长长长长长长长长长长，需要换行，参见 :func:`dramkit.gentools.isnull`
    参见属性 :py:attr:`name` 参见 :py:meth:`show`

    Attributes
    ----------
    name : str
        class nick name


    :py:attr:`name`


    参见
    :func:`dramkit.pystyles.dramkit_style.dramkit_funcsytle` 和
    :func:`dramkit.logtools.utils_logger.logger_show`

    See Also
    --------
    :func:`dramkit.logtools.utils_logger.logger_show`: 本程序中的function代码风格示例


    .. seealso::

        function :py:func:`dramkit.pystyles.dramkit_style.dramkit_funcsytle` 函数编码风格

    Note
    ----
    为什么Attributes标题没显示，怎么设置？attribute显示有点丑
    '''

    def __init__(self, name='DramkitDocStyle', parm2=None):
        '''
        初始化
        
        Parameters
        ----------
        name : str
            value for self.name
        parm2 : str, None
            self.vparm2

            
        属性可以写到这里
    
        Attributes
        ----------
        otherattr : str
            description of other attribute
        '''
        self.name = name
        self.parm2 = parm2

    def show(self, x=None):
        '''
        class中函数的注释参照function的注释规范

        Parameters
        ----------
        x : str
            string to print

        Examples
        --------
        >>> dds = DramkitDocStyle()
        >>> dds.show('example')
        DramkitDocStyle
        example
        '''

        print(self.name)
        print(x)
        
    def show2(self, y):
        '''
        | :py:meth:`show`
        | :py:meth:`dramkit.pystyles.dramkit_style.DramkitClassStyle.show`
        '''
        print(y)
        
    @property
    def account_id(self):
        '''账户id'''
        return 0
    
    @staticmethod
    def show3(x):
        '''staticmethod here'''
        print(x)
