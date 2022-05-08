# -*- coding: utf-8 -*-

try:
    import dramkit
except:
    import sys
    sys.path.append('../../../dramkit/')

from dramkit import simple_logger
from dramkit.iotools import find_files_include_str


if __name__ == '__main__':
    import time
    strt_tm = time.time()

    # logger = None
    logger = simple_logger()
    # target_str = 'get_next_nth_trade_date'
    # root_dir = '../../../HooFin/'
    target_str = 'utils_hoo'#'simple_logger'#'get_logger'
    root_dir = '../../../dramkit/'
    file_types = ['.py']
    

    files = find_files_include_str(target_str, root_dir, file_types,
                                   logger)


    print('used time: %ss.'%(round(time.time()-strt_tm, 6)))
