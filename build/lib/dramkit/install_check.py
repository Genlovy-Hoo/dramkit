# -*- coding: utf-8 -*-

from ._pkg_info import pkg_info


def install_check():
    '''
    检查是否成功安装dramkit
    
    若成功安装，会打印版本号和相关提示信息
    '''
    
    print(
        '''
        successfully installed, version: %s.
        for more information, use `%s.pkg_info`
        '''
        % (pkg_info['__version__'], pkg_info['__pkgname__']))
    

