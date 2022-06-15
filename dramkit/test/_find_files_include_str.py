# -*- coding: utf-8 -*-

try:
    import dramkit
except:
    import sys
    sys.path.append('../../../DramKit/')

from dramkit import simple_logger
from dramkit.iotools import find_files_include_str


if __name__ == '__main__':
    import time
    strt_tm = time.time()

    # logger = None
    logger = simple_logger()
    target_str = 'load_his_data'#'simple_logger'#'get_logger'
    root_dir = '../../../DramKit/'
    # target_str = 'get_regular'#'simple_logger'#'get_logger'
    # root_dir = '../../../ProjectsHundSun/金工平台/因子扩容改造/src/'
    # target_str = 'logging'#'simple_logger'#'get_logger'
    # root_dir = 'C:/ProgramData/Anaconda3/Lib/site-packages/tushare/'
    file_types = ['.py']

    files = find_files_include_str(target_str, root_dir, file_types,
                                   logger)


    print('used time: %ss.'%(round(time.time()-strt_tm, 6)))
