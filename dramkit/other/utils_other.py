# -*- coding: utf-8 -*-

import os
import subprocess
from utils_hoo.utils_io import load_csv, logger_show

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
