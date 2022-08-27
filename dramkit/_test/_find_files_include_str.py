# -*- coding: utf-8 -*-

from dramkit import simple_logger
from dramkit.iotools import find_files_include_str


if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    # logger = None
    logger = simple_logger()
    target_str = 'chinese_calendar'
    # target_str = "f'"
    root_dir = 'D:/Genlovy_Hoo/HooProjects/FinFactory/'
    # root_dir = 'D:/Genlovy_Hoo/HooProjects/DramKit/'
    file_types = ['.py']

    files = find_files_include_str(target_str, root_dir, file_types,
                                   logger)


    tr.used()
