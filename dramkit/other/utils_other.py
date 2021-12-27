# -*- coding: utf-8 -*-

import os
import subprocess
import pandas as pd
from dramkit.gentools import cut_df_by_con_val
from utils_hoo.utils_io import read_lines, load_csv, logger_show

def load_csv_ColMaxMin(csv_path, col='date', return_data=True, dropna=False,
                       **kwargs):
    '''
    获取csv_path历史数据col列（csv文件必须有col列）的最大值和最小值
    当return_data为True时返回最大值、最小值和df数据，为False时不返回数据（None）
    dropna设置在判断最大最小值之前是否删除col列的无效值
    **kwargs为load_csv可接受的参数
    '''
    data = load_csv(csv_path, **kwargs)
    if dropna:
        data.dropna(how='any', inplace=True)
    data.sort_values(col, ascending=True, inplace=True)
    col_Max, col_Min = data[col].iloc[-1], data[col].iloc[0]
    if return_data:
        return col_Max, col_Min, data
    else:
        return col_Max, col_Min, None


def load_text_multi(fpath, sep=',', encoding=None, del_first_col=False,
                    del_last_col=False, del_first_line=False, to_pd=True,
                    keep_header=True, logger=None):
    '''
    读取可能存在多个表纵向排列，且每个表列数不相同的文件，读取出每个表格

    Parameters
    ----------
    fpath: 文本文件路径
    sep: 字段分隔符，默认`,`
    encoding: 指定编码方式，默认不指定，不指定时会尝试以uft-8和gbk编码读取
    del_first_col: 是否删除首列，默认不删除
    del_last_col: 是否删除最后一列，默认否
    del_first_line: 是否删除首行，默认不删除
    to_pd: 是否输出为pandas.DataFrame，默认是
    keep_header: 输出为pandas.DataFrame时是否以首行作为列名，默认是
    logger: 日志记录器

    注：若del_first_line为True，则输出pandas.DataFrame没有列名

    Returns
    -------
    data: list或pandas.DataFrame
    '''

    if not os.path.exists(fpath):
        logger.warning('文件不存在，返回None：{}'.format(fpath))
        return None

    if encoding is not None:
        try:
            with open(fpath, 'r', encoding=encoding) as f:
                lines = f.readlines()
        except:
            lines = read_lines(fpath, logger=logger)
    else:
        lines = read_lines(fpath, logger=logger)

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

    def get_final_data(data):
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

    datas = [get_final_data(x) for x in datas]

    return datas


def install_pkg(pkg_name, version=None, upgrade=False, ignore_exist=False,
                logger=None):
    '''
    安装python库
    version格式: `==0.1.4`|`>1.0`|`<2.0`
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
