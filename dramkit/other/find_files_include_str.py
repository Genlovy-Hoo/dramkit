# -*- coding: utf-8 -*-

import os
import pandas as pd
from utils_hoo.utils_io import get_all_files, read_lines


def find_files_include_str(target_str, root_dir=None, file_types=None):
    '''
    在root_dir目录下的文件中，查找那些文件里面包含了target_str字符串，返回找到的文件
    路径列表
    file_types指定查找的文件后缀，all在所有文件中查找，None默人在常见的文本文件中查找
    '''
    if root_dir is None:
        root_dir = os.getcwd()
    if file_types is None:
        file_types = ['.py', '.txt', '.json', '.config', '.yml', '.cfg']
    if file_types == 'all':
        file_types = None
    all_files = get_all_files(root_dir, ext=file_types)
    files = []
    for fpath in all_files:
        lines = read_lines(fpath)
        for line in lines:
            try:
                if target_str in line:
                    files.append([fpath, line])
                    break
            except:
                if target_str.encode('gbk') in line:
                    files.append([fpath, line])
                    break
    files = pd.DataFrame(files, columns=['fpath', 'content'])
    return files


if __name__ == '__main__':
    import time
    strt_tm = time.time()

    target_str = 'simple_logger'
    root_dir = '../../'
    file_types = ['.py']

    files = find_files_include_str(target_str, root_dir, file_types)


    print('used time: %ss.'%(round(time.time()-strt_tm, 6)))
