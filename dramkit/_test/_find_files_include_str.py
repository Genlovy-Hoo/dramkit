# -*- coding: utf-8 -*-

from dramkit import simple_logger
from dramkit.iotools import find_files_include_str


if __name__ == '__main__':
    import time
    strt_tm = time.time()

    # logger = None
    logger = simple_logger()
    # target_str = 'utils_hoo'
    target_str = 'HooFin'
    # root_dir = 'D:/Genlovy_Hoo/HooProjects/FinFactory/'
    root_dir = 'D:/Genlovy_Hoo/HooProjects/DramKit/'
    file_types = ['.py', '.yml']

    files = find_files_include_str(target_str, root_dir, file_types,
                                   logger)


    print('used time: %ss.'%(round(time.time()-strt_tm, 6)))
