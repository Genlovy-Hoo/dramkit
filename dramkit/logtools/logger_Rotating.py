# -*- coding: utf-8 -*-

import os
import logging
from logging.handlers import RotatingFileHandler
from .utils_logger import remove_handlers
# from dramkit.logtools.utils_logger import remove_handlers


def get_logger(fpath=None, fmode='w', max_kb=1, nfile=3, screen_show=True):
    '''
    滚动日志记录(按文件大小)，将日志信息滚动保存在文件或在屏幕中打印

    Parameters
    ----------
    fapth : str, None
        日志文件路径，默认为None即不保存日志文件
    fmode : str
        'w'或'a'，取'w'时会覆盖原有日志文件（若存在的话），取'a'则追加记录
    max_kb : float
        单个日志文件大小限制，单位为kb
    nfile : int
        最多备份文件个数
    screen_show : bool
        是否在控制台打印日志信息，默认打印

    Note
    ----
    ``fpath`` 和 ``screen_show`` 必须有至少一个为真

    Returns
    -------
    logger : logging.Logger
        日志记录器

    See Also
    --------
    常规日志记录: :func:`dramkit.logtools.logger_general.get_logger`

    日志文件按时间滚动: :func:`dramkit.logtools.logger_timedrotating.get_logger`
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
        if fmode == 'w' and os.path.exists(fpath):
            # 先删除原有日志文件
            fdir = os.path.dirname(os.path.realpath(fpath))
            fname = os.path.basename(fpath)
            fpaths = [x for x in os.listdir(fdir) if fname in x]
            for f in fpaths:
                os.remove(os.path.join(fdir, f))
        # 日志文件保存，FileHandler
        file_logger = RotatingFileHandler(fpath, mode=fmode,
                                          maxBytes=max_kb*1024,
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
    from dramkit.logtools.utils_logger import close_log_file

    log_path = './test/log_test2.log'
    logger = get_logger(fpath=log_path, fmode='w', max_kb=1, nfile=3,
                        screen_show=True)

    logger.info('Log start here ********************************************')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')

    logger.info('Log start here ---------------------------------------------')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')

    logger.info('Log start here #############################################')
    logger.debug('Do something.')
    logger.warning('Something maybe fail.')
    logger.error('Some error find here.')
    logger.critical('Program crashed.')
    logger.info('Finish')

    close_log_file(logger)
