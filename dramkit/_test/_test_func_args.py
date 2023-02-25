# -*- coding: utf-8 -*-

import sys, inspect, re
from dramkit.gentools import get_func_arg_info


if __name__ == '__main__':
    
    def func(a1, a2, /, b1, b2, *, c1, c2=1):
        '''
        | a1, a2: 只接受位置参数
        | b1, b2: 接受位置参数和关键字参数
        | c1, c2: 只接受关键字参数
        | 示例调用: func(1, 2, 3, b2=4, c1=5, c2=6)
        | 参考：https://www.bilibili.com/video/av338046022/
        '''
        print(a1, a2, b1, b2, c1, c2)        
    # func(1, 2, 3, 4, 5, 6) # 报错
    func(1, 2, 3, b2=4, c1=5, c2=6)
    print(get_func_arg_info(func))
    
    
    def show_vars(a, *args, b=1, **kwargs):
        print(locals())
        x = 1
        print(locals())
    show_vars(1, 2, 3)
    show_vars(1, 2, 3, y=3)
    
    
    def func2(x, y, z=1, *args, **kwargs):
        fname = sys._getframe().f_code.co_name
        exec('d = 1')
        c = 2
        print('c:', c)
        argnames = eval('%s.__code__.co_varnames'%fname)
        print('变量名称列表：', argnames)
    func2(2, 3)
    
    
    def get_fund_self_name():
        # https://dandelioncloud.cn/article/details/1482263594321641474/
        now_func_name = sys._getframe().f_code.co_name
        now_file_name = sys._getframe().f_code.co_filename
        print('当前函数名称：', now_func_name)
        print('当前文件名称：', now_file_name)
    get_fund_self_name()
    
    
    def get_varname(*args, **kwargs):
        '''
        | 将变量名转化为同名字符串
        | https://www.zhihu.com/question/42768955
        | https://www.cnblogs.com/yoyoketang/p/9231320.html
        '''
        now_func_name = sys._getframe().f_code.co_name
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            # print('line:', line)
            m = re.search(r'\b'+now_func_name+r'\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)      
    print('参数名称：', get_varname(233))
    shit = 233
    print('参数名称：', get_varname(shit))
    vstr = 'shit=233'
    print('参数名称：', get_varname(exec(vstr)))
    x = 'shit'
    print('参数名称：', get_varname(eval(x)))
    
    