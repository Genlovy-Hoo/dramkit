# -*- coding: utf-8 -*-

from dramkit.iotools import get_all_paths, read_lines, write_txt


def replace_endblank_in_file(fpath, func_new_path=None, logger=None,
                             encoding=None, encoding_new='utf-8'):
    '''
    替换指定文本文件中每一行行尾部的空白，并存为新文件

    Parameters
    ----------
    fpath : str
        待替换目标文本文件路径
    func_new_path : function
        新文件的名称由func_new_path函数给出，fpath_new = func_new_path(fpath)
    logger : None, logging.Logger
        日志记录器
    encoding : None, str
        原文件编码方式
    encoding_new : str
        新文件编码方式
    '''

    def _func_new_path(x):
        ftype = '.' + x.split('.')[-1]
        path_ = x.replace(ftype, '')
        return path_ + '_noendblank' + ftype

    if func_new_path is None:
        func_new_path = _func_new_path

    # 新路径
    fpath_new = func_new_path(fpath)

    # 读取文件内容并替换尾部空白
    lines = read_lines(fpath, encoding=encoding, logger=logger)
    lines_new = []
    for line in lines:
        lines_new.append(line.rstrip())
    while lines_new[-1] == '':
        lines_new.pop()

    # 写入新文件
    write_txt(lines_new, fpath_new, encoding=encoding_new)


if __name__ == '__main__':
    from dramkit import TimeRecoder
    tr = TimeRecoder()

    root_dir = './'
    files_types = ['.py_']
    all_files = get_all_paths(root_dir, ext=files_types)

    # func_new_path = lambda x: x
    func_new_path = None
    encoding_new = 'utf-8'
    for fpath in all_files:
        replace_endblank_in_file(fpath, func_new_path,
                                 encoding_new=encoding_new)


    tr.used()
