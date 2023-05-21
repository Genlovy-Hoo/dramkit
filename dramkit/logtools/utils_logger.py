# -*- coding: utf-8 -*-

import os
import logging
import datetime
from pandas import isnull


# 日志格式
# formatter = logging.Formatter(
# '''%(asctime)s -%(filename)s[line: %(lineno)d] -%(levelname)s:
# --%(message)s''')
# formatter = logging.Formatter(
# '''%(asctime)s -%(name)s[line: %(lineno)d] -%(levelname)s:
# --%(message)s''')
# formatter = logging.Formatter(
# '''%(asctime)s -%(levelname)s:
# --%(message)s''')
formatter = logging.Formatter(
'''%(message)s
    [%(levelname)s: %(asctime)s]''')


def logger_show(log_str, logger=None, level='info'):
    '''
    显示|记录日志信息

    parameters
    ----------
    log_str : str
        日志内容字符串
    logger : logging.Logger, None, nan, False
        - 若为False，则不显示|记录日志信息
        - 若为None或nan，则用print打印log_str
        - 若为logging.Logger对象，则根据level设置进行日志显示|记录
    level : str
        支持'info', 'warn', 'error'三个日志级别


    .. todo::
        - 添加更多level和设置
    '''
    if isnull(logger):
        print(log_str)
        print('    [time: {}]'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
    elif logger is False:
        return
    elif isinstance(logger, logging.Logger):
        if level == 'info':
            logger.info(log_str)
        elif level in ['warn', 'warning']:
            logger.warning(log_str)
        elif level in ['error', 'err']:
            logger.error(log_str, exc_info=True)
        else:
            raise ValueError('未识别的日志级别设置！')
    else:
        raise ValueError('未识别的logger！')


def close_log_file(logger):
    '''关闭日志记录器logger中的文件流，返回logger'''
    if isnull(logger) or logger is False:
        return logger
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    return logger


def remove_handlers(logger):
    '''
    关闭并移除logger中已存在的handlers，返回logger

    .. note::
        貌似必须先把FileHandler close并remove之后，再remove其它handler
        才能完全remove所有handlers，原因待查（可能由于FileHandler
        是StreamHandler的子类的缘故？）
    '''
    # # 逐个移除
    # for h in logger.handlers:
    #     if isinstance(h, logging.FileHandler):
    #         h.close()
    #         logger.removeHandler(h)
    # for h in logger.handlers:
    #     logger.removeHandler(h)
    # 直接清空
    logger.handlers = []
    return logger


def _get_level(level=None):
    '''获取显示级别'''
    if level in [None, 'debug', 'DEBUG', logging.DEBUG]:
        return logging.DEBUG
    elif level in ['info', 'INFO', logging.INFO]:
        return logging.INFO
    elif level in ['warning', 'WARNING', 'warn', 'WARN', logging.WARNING]:
        return logging.WARNING
    elif level in ['error', 'ERROR', 'err', 'ERR', logging.ERROR]:
        return logging.ERROR
    elif level in ['critical', 'CRITICAL', logging.CRITICAL]:
        return logging.CRITICAL
    else:
        raise ValueError('level参数设置有误，请检查！')


def set_level(logger, level=None):
    '''设置日志显示基本'''
    logger.setLevel(_get_level(level))
    return logger


def _pre_get_logger(fpath, screen_show, logname, level):
    if fpath is None and not screen_show:
        raise ValueError('`fpath`和`screen_show`必须至少有一个为真！')
    # 准备日志记录器logger
    if logname is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(logname)
    # 显示级别
    logger = set_level(logger, level=level)
    # 预先删除logger中已存在的handlers
    logger = remove_handlers(logger)
    return logger
        
        
def make_path_dir(fpath):
    '''若fpath所指文件夹路径不存在，则新建之'''
    if isnull(fpath):
        return
    dir_path = os.path.dirname(fpath)
    if not os.path.exists(dir_path) and len(dir_path) > 0:
        os.makedirs(dir_path)
