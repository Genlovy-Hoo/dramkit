# -*- coding: utf-8 -*-

from dramkit import simple_logger
from dramkit.iotools import find_files_include_str


if __name__ == '__main__':
    import time
    strt_tm = time.time()

    # logger = None
    logger = simple_logger()
    target_str = 'simple_logger'
    root_dir = '../../'
    file_types = None
    

    files = find_files_include_str(target_str, root_dir, file_types,
                                   logger)


    print('used time: %ss.'%(round(time.time()-strt_tm, 6)))
