# -*- coding: utf-8 -*-

import logging
from logging.handlers import TimedRotatingFileHandler
from .utils_logger import remove_handlers
# from dramkit.logtools.utils_logger import remove_handlers


def get_logger(fpath=None, when='S', interval=3, nfile=3,
               screen_show=True):
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

    if fpath is None and not screen_show:
        raise ValueError('`fpath`和`screen_show`必须至少有一个为真！')

    # 准备日志记录器logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 预先删除logger中已存在的handlers
    logger = remove_handlers(logger)

    # 日志格式
    formatter = logging.Formatter(
    '''%(asctime)s -%(filename)s[line: %(lineno)d] -%(levelname)s:
    --%(message)s''')

    if fpath is not None:
        # 日志文件保存，FileHandler
        file_logger = TimedRotatingFileHandler(fpath,
                                               when=when,
                                               interval=interval,
                                               backupCount=nfile)
        file_logger.setLevel(logging.DEBUG)
        file_logger.setFormatter(formatter)
        logger.addHandler(file_logger)

    if screen_show:
        # 控制台打印，StreamHandler
        console_logger = logging.StreamHandler()
        console_logger.setLevel(logging.DEBUG)
        console_logger.setFormatter(formatter)
        logger.addHandler(console_logger)

    return logger


if __name__ == '__main__':
    import time
    from dramkit.logtools.utils_logger import close_log_file

    log_path = './test/log_test3.log'
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
