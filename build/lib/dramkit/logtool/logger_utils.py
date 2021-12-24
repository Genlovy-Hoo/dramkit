# -*- coding: utf-8 -*-

import logging
from utils_hoo.utils_general import isnull


def logger_show(log_str, logger=None, level='info'):
    '''
    todo: 添加更多level和设置
    
    显示|记录日志信息
    
    parameters
    ----------
    log_str: str，日志内容
    logger:
        若logger为False，则不显示|记录日志信息
        若logger为None或无效值，则用print打印日志信息
        若logger为正常loggering对象，则根据level进行日志显示|记录
    level: 支持'info', 'warn', 'error'三个级别
    '''    
    if isnull(logger):
        print(log_str)
    elif logger == False:
        return
    elif isinstance(logger, logging.Logger):
        if level == 'info':
            logger.info(log_str)
        elif level in ['warn', 'warning']:
            logger.warning(log_str)
        elif level in ['error', 'err']:
            logger.error(log_str, exc_info=True)
        else:
            raise Exception('未识别的日志级别设置！')
    else:
        raise Exception('未识别的logger！')


def close_log_file(logger):
    '''关闭日志记录器logger中的文件流'''
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    return logger
            
        
def remove_handlers(logger):
    '''
    关闭并移除logger中已存在的handlers
    注：这里貌似必须先把FileHandler close并remove之后，
       再remove其它handler才能完全remove所有handlers，原因待查（可能由于FileHandler
       是StreamHandler的子类的缘故）
    '''
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
    for h in logger.handlers:
            logger.removeHandler(h)
    return logger
