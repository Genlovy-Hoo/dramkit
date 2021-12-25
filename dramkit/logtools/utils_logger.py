# -*- coding: utf-8 -*-

import logging
from dramkit.gentools import isnull


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
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
    for h in logger.handlers:
            logger.removeHandler(h)
    return logger
