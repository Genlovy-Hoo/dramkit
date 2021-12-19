# -*- coding: utf-8 -*-

from ._pkg_info import pkg_info


def install_check():
    print(
        '''
        successfully installed, version: %s.
        for more information, use `%s.pkg_info`
        '''
        % (pkg_info['__version__'], pkg_info['__pkgname__']))
    

