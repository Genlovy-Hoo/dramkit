# -*- coding: utf-8 -*-

from dramkit import simple_logger
from dramkit.iotools import find_dir_include_str


if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    # logger = None
    logger = simple_logger()
    
    target_str = 'chncal'
    # target_str = "f'"
    root_dir = 'D:/Genlovy_Hoo/HooProjects/FinFactory/'
    # root_dir = 'D:/Genlovy_Hoo/HooProjects/DramKit/'
    file_types = ['.py']

    files = find_dir_include_str(target_str,
                                 root_dir=root_dir,
                                 file_types=file_types,
                                 logger=logger,
                                 return_all_find=True)
    
    
    target_str = r'\.loc\[.*:'
    root_dir = 'D:/Genlovy_Hoo/HooProjects/DramKit/'
    file_types = ['.py']
    files1 = find_dir_include_str(target_str,
                                  root_dir=root_dir,
                                  file_types=file_types,
                                  re_match=True, logger=logger)


    tr.used()
