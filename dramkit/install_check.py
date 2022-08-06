# -*- coding: utf-8 -*-

from ._pkg_info import pkg_info
from .logtools.logger_general import get_logger
from .logtools.utils_logger import logger_show


suscess_str = '''successfully installed, version: %s.
      for more information, use `%s.pkg_info`
''' % (pkg_info['__version__'], pkg_info['__pkgname__'])


def install_check():
    '''
    检查是否成功安装dramkit

    若成功安装，会打印版本号和相关提示信息
    '''
    
    logger = get_logger()
    
    try:
        from dramkit import load_csv
        logger_show(suscess_str, logger, 'info')
    except:
        import traceback
        print(traceback.format_exc())
        print('未成功安装dramkit, 请检查！')