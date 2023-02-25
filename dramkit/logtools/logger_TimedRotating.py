# -*- coding: utf-8 -*-

import logging
from logging.handlers import TimedRotatingFileHandler
from dramkit.logtools.utils_logger import (
                                   _pre_get_logger,
                                   _get_level,
                                   formatter,
                                   make_path_dir)


def get_logger(fpath=None, when='M', interval=3, nfile=3,
               logname=None, level=None, screen_show=True):
    '''
    滚动日志记录（按时间），将日志信息滚动保存在文件或在屏幕中打印

    Parameters
    ----------
    fapth : str, None
        日志文件路径，默认为None即不保存日志文件
    when : str
        回滚时间单位:
        ``S`` 秒、``M`` 分、``H`` 小时、``D`` 天、``W`` 星期、``midnight`` 午夜 等
    interval : int
        滚动周期，单位由 ``when`` 指定
    nfile : int
        最多备份文件个数
    screen_show : bool
        是否在控制台打印日志信息，默认打印

        .. note:: ``fpath`` 和 ``screen_show`` 必须有至少一个为真


    :returns: `logging.Logger` - 日志记录器

    See Also
    --------
    常规日志记录: :func:`dramkit.logtools.logger_general.get_logger`

    日志文件按大小滚动: :func:`dramkit.logtools.logger_rotating.get_logger`
    '''

    make_path_dir(fpath)
    logger = _pre_get_logger(fpath, screen_show, logname, level)

    if fpath is not None:
        # 日志文件保存，FileHandler
        file_logger = TimedRotatingFileHandler(fpath,
                                               when=when,
                                               interval=interval,
                                               backupCount=nfile)
        file_logger.setLevel(_get_level(level))
        file_logger.setFormatter(formatter)
        logger.addHandler(file_logger)

    if screen_show:
        # 控制台打印，StreamHandler
        console_logger = logging.StreamHandler()
        console_logger.setLevel(_get_level(level))
        console_logger.setFormatter(formatter)
        logger.addHandler(console_logger)

    return logger


if __name__ == '__main__':
    import time
    from dramkit.logtools.utils_logger import close_log_file

    log_path = './_test/log_test3.log'
    logger = get_logger(fpath=log_path, when='S', interval=3,
                        screen_show=True)

    count = 0
    while count < 10:
        logger.info('Log start here *****************************************')
        logger.debug('Do something.')
        logger.warning('Something maybe fail.')
        logger.error('Some error find here.')
        logger.critical('Program crashed.')
        logger.info('Finish')

        time.sleep(2)
        count += 1

    close_log_file(logger)
