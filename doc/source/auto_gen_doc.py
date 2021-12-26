# -*- coding: utf-8 -*-

'''
自动生成rst文档
注：目前只考虑.py脚本，没考虑二进制文件

todo:
    init中现有__all__限制的情况
    名称映射中文乱码问题
'''

import os
import inspect


PKG_NAME = 'dramkit'
DOC_TITLE = 'DramKit'
SRC_DIR = '../../%s/'%PKG_NAME
PKG_DIR = '../../../%s'%PKG_NAME
INCLUDE_ = False

# 名称映射，中文有乱码问题
NAME_MAP = {}

# 目录顺序
ORDER = {}


import sys
sys.path.append(PKG_DIR)
from dramkit.logtools.utils_logger import logger_show
from dramkit.logtools.logger_general import get_logger
from dramkit.iotools import read_lines, write_txt, get_all_files


def namemap(name):
    if name in NAME_MAP:
        return NAME_MAP[name]
    return name


def get_sub_pkgs():
    '''查找所有子package名称列表'''
    sub_pkgs = os.listdir(SRC_DIR)
    sub_pkgs = [x for x in sub_pkgs if os.path.isdir(SRC_DIR+x)]
    ignores = ['__pycache__', '.ipynb_checkpoints', '.pylint.d', 'test']
    sub_pkgs = [x for x in sub_pkgs if x not in ignores]
    return sub_pkgs


def get_pys(sub_pkg, include_=False):
    '''
    读取sub_pkg指定名称的子package下所有.py脚本路径
    include_设置返回结果是否包含以'_'开头的被保护脚本
    '''
    pys = os.listdir('%s%s'%(SRC_DIR, sub_pkg))
    pys = [x for x in pys if x[-3:] == '.py']
    pys = [x for x in pys if '__init__.py' not in x]
    if not include_:
        pys = [x for x in pys if x[0] != '_']
    if sub_pkg == '':
        pys = ['%s%s'%(SRC_DIR, x) for x in pys]
    else:
        pys = ['%s%s/%s'%(SRC_DIR, sub_pkg, x) for x in pys]
    pys = [x for x in pys if not os.path.isdir(x)]
    return pys


def get_class_funcs(pypath, include_=False):
    '''
    给定py脚本文件路径，读取脚本中定义的所有class和function名称列表
    include_设置返回结果是否包含私有方法（定义时以'_'开头）
    返回结果形如: [class1, class2, ...], [func1, func2, ...]
    注意：默认所有类和函数均在首行无缩进位置定义
    '''
    lines = read_lines(pypath)
    classs, methods = [], []
    for line in lines:
        if line[:4] == 'def ':
            if line[:5] == 'def _':
                if include_:
                    methods.append(line)
                else:
                    pass
            else:
                methods.append(line)
        elif line[:6] == 'class ':
            classs.append(line)
        else:
            continue
    methods = [x.split('(')[0][4:] for x in methods]
    classs = [x.split('(')[0][6:] for x in classs]
    return classs, methods


def get_class_funcs_init():
    '''获取package根目录下__init__中导入的类和函数列表'''
    exec('import %s'%PKG_NAME)
    initclass = inspect.getmembers(eval(PKG_NAME),
                                   inspect.isclass)
    initclass = [x[0] for x in initclass]
    initfuncs = inspect.getmembers(eval(PKG_NAME),
                                   inspect.isfunction)
    initfuncs = [x[0] for x in initfuncs]
    return initclass, initfuncs


def write_subpkg_rst(sub_pkg, include_=False):
    '''读取sub_pkg子库的内容并编写相应rst文档'''

    # .rst文档
    if sub_pkg == '':
        frst = PKG_NAME + '.rst'
        # package名称
        rst_lines = [PKG_NAME]
        rst_lines.append('=' * (len(PKG_NAME)+1))
        rst_lines.append('')
    else:
        frst = '%s.%s.rst'%(PKG_NAME, sub_pkg)
        # 子package名称
        name_ = namemap(sub_pkg)
        rst_lines = [name_]
        rst_lines.append('=' * (len(name_)+1))
        rst_lines.append('')

    pys = get_pys(sub_pkg, include_=include_)
    pys.sort()

    for fpy in pys:
        module = os.path.basename(fpy)[:-3]

        # module名称
        name_ = namemap(module)
        rst_lines.append(name_)
        rst_lines.append('-' * (len(name_)+1))
        rst_lines.append('')

        # module完整路径
        if sub_pkg == '':
            mdlfull = '.'.join([PKG_NAME, module])
        else:
            mdlfull = '.'.join([PKG_NAME, sub_pkg, module])
        rst_lines.append('.. automodule:: '+mdlfull)
        rst_lines.append('')
        rst_lines.append('.. currentmodule:: '+mdlfull)
        rst_lines.append('')

        clas, funcs = get_class_funcs(fpy, include_=include_)
        clas.sort()
        funcs.sort()

        # 添加class
        for clss in clas:
            name_ = namemap(clss)
            rst_lines.append(name_)
            rst_lines.append('^' * (len(name_)+1))
            rst_lines.append('')
            rst_lines.append('.. autoclass:: '+mdlfull+'.'+clss)
            for x in [':members:', ':undoc-members:', ':show-inheritance:']:
                rst_lines.append('    ' + x)
            rst_lines.append('')

        # 添加method
        for fnc in funcs:
            name_ = namemap(fnc)
            rst_lines.append(name_)
            rst_lines.append('^' * (len(name_)+1))
            rst_lines.append('')
            rst_lines.append('.. autofunction:: '+mdlfull+'.'+fnc)
            rst_lines.append('')

    if rst_lines[-1] == '':
        rst_lines.pop()

    write_txt(rst_lines, frst)


if __name__ == '__main__':
    import subprocess

    import time
    strt_tm = time.time()

    # __init__.py临时改名
    initfiles = get_all_files(SRC_DIR, ext=['__init__.py'])
    initfiles_tmp = [x+'tmp' for x in initfiles]
    for k in range(len(initfiles)):
        os.rename(initfiles[k], initfiles_tmp[k])

    # 子package的rst文档生成
    sub_pkgs = get_sub_pkgs()
    sub_pkgs.sort()
    sub_pkgs.insert(0, '')
    for sub_pkg in sub_pkgs:
        write_subpkg_rst(sub_pkg, include_=INCLUDE_)

    # 生成index.rst
    idxrst = 'index.rst'
    title = "Welcome to %s's documentation!"%DOC_TITLE
    rst_lines = [title]
    rst_lines.append('=' * (len(title)+1))
    rst_lines.append('')
    rst_lines.append('.. toctree::')
    rst_lines.append('   :maxdepth: 2')
    rst_lines.append('   :caption: Contents:')
    rst_lines.append('')
    rst_lines.append('   %s <%s>'%(PKG_NAME, PKG_NAME))
    for x in sub_pkgs[1:]:
        rst_lines.append('   %s <%s.%s>'%(x, PKG_NAME, x))
    rst_lines.append('')
    idxtitle = 'Indices and tables'
    rst_lines.append(idxtitle)
    rst_lines.append('=' * (len(idxtitle)+1))
    rst_lines.append('')
    rst_lines.append('* :ref:`genindex`')
    rst_lines.append('* :ref:`modindex`')
    write_txt(rst_lines, idxrst)

    # 执行.\make html生成文档
    nowpath = os.getcwd()
    docpath = os.path.abspath('../')
    os.chdir(docpath)

    # os.system('.\make html') # 无返回
    # subprocess.call('.\make html', shell=True) # 无返回
    # subprocess.check_call('.\make html', shell=True) # 无返回
    makeinfo = subprocess.check_output('.\make html', shell=True,
                                       stderr=subprocess.STDOUT)
    makeinfo = makeinfo.decode('gbk')
    makeinfo = makeinfo.replace('\r\n', '\n')
    logger_show(makeinfo, get_logger('./source/makeinfo.log'))

    # 将__init__文件名改回来
    os.chdir(nowpath)
    for k in range(len(initfiles)):
        os.rename(initfiles_tmp[k], initfiles[k])
    # 强制修改
    initfiles = get_all_files(SRC_DIR, ext=['__init__.pytmp'])
    initfiles_tmp = [x[:-3] for x in initfiles]
    for k in range(len(initfiles)):
        os.rename(initfiles[k], initfiles_tmp[k])


    print('used time: %ss.'%round(time.time()-strt_tm, 6))
