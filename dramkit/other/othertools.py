# -*- coding: utf-8 -*-

'''
Uncommonly used utility functions
'''

import os
import subprocess
import pandas as pd
from dramkit.gentools import cut_df_by_con_val
from dramkit.iotools import read_lines, load_csv, logger_show


def get_csv_colmaxmin(csv_path, col, skipna=True, return_data=True,
                      ascending=None, **kwargs):
    '''
    获取指定csv文件中指定列的最大值和最小值

    Parameters
    ----------
    csv_path : str
        csv数据文件路径
    col : str
        指定列名
    skipna : bool
        计算max和min的时候设置是否skipna
    return_data : bool
        为True时返回最大值、最小值和df数据，为False时不返回数据(替换为None)
    ascending : None, bool
        返回数据按col列排序: None不排序, True升序, False降序
    **kwargs :
        :func:`dramkit.iotools.load_csv` 函数接受的参数

    Returns
    -------
    col_max :
        col列最大值
    col_min :
        col列最小值
    data : None, pandas.DataFrame
        返回数据
    '''
    data = load_csv(csv_path, **kwargs)
    col_max = data[col].max(skipna=skipna)
    col_min = data[col].min(skipna=skipna)
    if return_data:
        if ascending is not None:
            data.sort_values(col, ascending=ascending, inplace=True)
        return col_max, col_min, data
    else:
        return col_max, col_min, None


def load_text_multi(fpath, sep=',', encoding=None, del_first_col=False,
                    del_last_col=False, del_first_line=False, to_pd=True,
                    keep_header=True, logger=None):
    '''
    读取可能存在多个表纵向排列，且每个表列数不相同的文件，读取出每个表格
    (中金所持仓排名数据中存在这种情况)

    Parameters
    ----------
    fpath : str
        文本文件路径
    sep : str
        字段分隔符，默认`,`
    encoding : None, str
        指定编码方式，为None时会尝试以uft-8和gbk编码读取
    del_first_col : bool
        是否删除首列，默认不删除
    del_last_col : bool
        是否删除最后一列，默认否
    del_first_line : bool
        是否删除首行，默认不删除

        .. note:: 若del_first_line为True，则输出pandas.DataFrame没有列名
    to_pd : bool
        是否输出为pandas.DataFrame，默认是
    keep_header : bool
        输出为pandas.DataFrame时是否以首行作为列名，默认是
    logger : logging.Logger, None
        日志记录器


    :returns: `list` - 返回读取的数据列表，元素为pandas.DataFrame或list
    '''

    if not os.path.exists(fpath):
        logger_show('文件不存在，返回None：%s'%fpath, logger, 'warn')
        return None
    
    lines = read_lines(fpath, encoding=encoding, logger=logger)

    data = []
    lens = []
    for line in lines:
        line = str(line)
        line = line.strip()
        if line == '':
            continue
        line = line.split(sep)
        if del_first_col:
            line = line[1:]
        if del_last_col:
            line = line[:-1]
        data.append(line)
        lens.append(len(line))

    tmp = pd.DataFrame({'len': lens})
    tmp['idx'] = range(0, tmp.shape[0])
    tmps = cut_df_by_con_val(tmp, 'len')
    start_end_idxs = [(x['idx'].iloc[0], x['idx'].iloc[-1]) for x in tmps]

    datas = [data[idx1:idx2+1] for idx1, idx2 in start_end_idxs]

    def _get_final_data(data):
        '''组织数据输出格式'''
        if del_first_line:
            data = data[1:]
            if to_pd:
                data = pd.DataFrame(data)
        else:
            if to_pd:
                if keep_header:
                    cols = data[0]
                    data = pd.DataFrame(data[1:])
                    data.columns = cols
                else:
                    data = pd.DataFrame(data)
        return data

    datas = [_get_final_data(x) for x in datas]

    return datas


def install_pkg(pkg_name, version=None, upgrade=False,
                ignore_exist=False, logger=None):
    '''
    安装python库

    Parameters
    ----------
    pkg_name : str
        库名称
    version : 安装版本
        格式: '==0.1.4'|'>1.0'|'<2.0'
    upgrade : bool
        是否更新
    ignore_exist : bool
        是否忽略已安装的包(有时候已安装的包无法卸载导致不能安装, 设置为True强制安装)
    logger : None, logging.Logger
        日志记录器


    .. todo::
        更多安装选项设置，如是否跳过依赖包等
    '''

    if ignore_exist:
        ignr = '--ignore-installed'

    if pkg_name[-4:] == '.whl' and os.path.exists(pkg_name):
        cmd_str = 'pip install {} {}'.format(os.path.abspath(pkg_name),
                                             ignr)
    else:
        if version is not None:
            cmd_str = 'pip install {}{} {}'.format(
                                        pkg_name, version, ignr)
        else:
            upgrade_str = '--upgrade' if upgrade else ''
            cmd_str = 'pip install {} {} {}'.format(
                                        upgrade_str, pkg_name, ignr)

    logger_show('安装{} ...'.format(pkg_name), logger)

    # os.system(cmd_str) # windows下会闪现cmd界面
    subprocess.call(cmd_str, shell=True)


if __name__ == '__main__':
    fpath = '../test/中金所持仓排名_IF20100416.csv'
    datas = load_text_multi(fpath)
    
    csv_path = '../test/510050_daily_pre_fq.csv'
    datemax, datemin, data = get_csv_colmaxmin(csv_path, 'date',
                                               return_data=False)