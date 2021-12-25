# -*- coding: utf-8 -*-

import os
import json
import pickle
import shutil
import zipfile
import subprocess
import pandas as pd
from .logtools.utils_logger import logger_show
from .gentools import simple_logger, isnull, cut_df_by_con_val
# from dramkit.logtools.logger_utils import logger_show
# from dramkit.gentools import simple_logger, isnull, cut_df_by_con_val


def copy_file():
	'''复制文件'''
	raise NotImplementedError


def copy_dir():
    '''复制文件夹'''
    raise NotImplementedError


def extract_7z(zip_path, save_dir):
    '''
    7z命令解压文件
    '''
    raise NotImplementedError


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


def rename_files_in_dir(dir_path, func_rename):
    '''
    对dir_path中的文件进行批量重命名
    name_new = func_rename(name)为重命名规则函数
    '''

    files = os.listdir(dir_path)
    for name in files:
        name_new = func_rename(name)
        old_path = os.path.join(dir_path, name)
        new_path = os.path.join(dir_path, name_new)
        os.rename(old_path, new_path)


def load_json(fpath, encoding=None, logger=None):
    '''读取json格式文件file'''

    if isnull(logger):
        logger = simple_logger()

    if not os.path.exists(fpath):
        logger.warning('文件不存在，返回None：{}'.format(fpath))
        return None

    try:
        with open(fpath, 'r', encoding=encoding) as f:
            data_json = json.load(f)
    except:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
        except:
            try:
                with open(fpath, 'r', encoding='gbk') as f:
                    data_json = json.load(f)
            except:
                raise IOError(
                    '读取{}出错，请检查文件（如编码或文件末尾多余字符等问题）！'.format(fpath))

    return data_json


def write_json(data, fpath, encoding=None):
    '''把data（dict）写入json格式文件file'''
    with open(fpath, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def zip_fpath_7z(fpath, zip_path=None, mode='zip', pwd=None,
                 keep_zip_new=True):
    '''
    7z命令压缩单个文件（夹）

    fpath: 待压缩文件(夹)路径
    zip_path: 压缩文件保存路径，若为None，则为fpath路径加后缀
    mode: 压缩文件后缀，可选['7z', 'zip']
    pwd: 密码字符串
    keep_zip_new: 为True时若zip_path已经存在, 则会先删除已经存在的再重新创建新压缩文件
    '''

    fpath = os.path.abspath(fpath) # 绝对路径

    if isnull(zip_path):
        zip_path = fpath + '.zip'
    else:
        zip_path = os.path.abspath(zip_path)

    if os.path.exists(zip_path) and keep_zip_new:
        os.remove(zip_path)

    md_str = ' -t' + mode

    if isnull(pwd):
        pwd = ''
    else:
        pwd = ' -p' + str(pwd)

    cmd_str = '7z a ' + zip_path + ' ' + fpath + md_str + pwd

    # os.system(cmd_str) # windows下会闪现cmd界面
    subprocess.call(cmd_str, shell=True)


def zip_fpaths_7z(zip_path, fpaths, mode='zip', pwd=None, keep_zip_new=True):
    '''
    7z命令压缩多个文件（夹）列表files到压缩文件zip_path
    注: files中的单个文件(不是文件夹)在压缩包中不会保留原来的完整路径,
        所有单个文件都会在压缩包根目录下

    zip_path: 压缩文件保存路径
    fpaths: 压缩文件路径列表，fpaths太长的时候会出错
    mode: 压缩文件后缀，可选['7z', 'zip']
    pwd: 密码字符串
    keep_zip_new: 为True时若zip_path已经存在, 则会先删除已经存在的再重新创建新压缩文件
    '''

    md_str = ' -t' + mode

    if isnull(pwd):
        pwd = ''
    else:
        pwd = ' -p' + str(pwd)

    if os.path.exists(zip_path) and keep_zip_new:
        os.remove(zip_path)

    fpaths_str = ' '.join([os.path.abspath(x) for x in fpaths])

    cmd_str = '7z a ' + zip_path + ' ' + fpaths_str + md_str + pwd

    # os.system(cmd_str) # windows下会闪现cmd界面
    subprocess.call(cmd_str, shell=True)


def zip_fpath(fpath, zip_path=None):
    '''
    zipfile压缩单个文件（夹）

    fpath: 待压缩文件夹路径(应为相对路径)
    zip_path: 压缩文件保存路径，若为None，则为fpath路径加后缀
    '''

    if isnull(zip_path):
        if os.path.isdir(fpath) and fpath[-1] == '/':
            zip_path = fpath[:-1] + '.zip'
        else:
            zip_path = fpath + '.zip'

    if os.path.isfile(fpath): # fpath为文件
        zip_files(zip_path, [fpath])
    elif os.path.isdir(fpath): # fpath为文件夹
        fpaths = get_all_files(fpath)
        zip_files(zip_path, fpaths)


def zip_fpaths(zip_path, fpaths):
    '''
    压缩路径列表fpaths(可为文件也可为文件夹, 应为相对路径)到zip_path
    zip_path：zip压缩包保存路径
    fpaths：需要压缩的路径列表(应为相对路径, 可为文件也可为文件夹)
    '''
    all_paths = []
    for fpath in fpaths:
        if os.path.isfile(fpath):
            all_paths.append(fpath)
        elif os.path.isdir(fpath):
            all_paths += get_all_files(fpath)
    zip_files(zip_path, all_paths)


def zip_files(zip_path, fpaths, keep_ori_path=True):
    '''
    使用zipfile打包为.zip文件
    zip_path：zip压缩包保存路径
    fpaths：需要压缩的文件(不是能文件夹)路径列表(应为相对路径)
    keep_ori_path: 若为True, 则压缩文件会保留fpaths中文件的原始路径
                   若为False, 则fpaths中所有文件在压缩文件中都在统一根目录下
    '''
    f = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    if keep_ori_path:
        for fpath in fpaths:
            f.write(fpath)
    else:
        for fpath in fpaths:
            file = os.path.basename(fpath)
            f.write(fpath, file)
    f.close()


def get_all_files(dir_path, ext=None):
    '''
	获取dir_path文件夹及其子文件夹中所有的文件路径
	ext指定文件后缀列表
	'''
    assert (ext is None or isinstance(ext, list))
    fpaths = []
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            if ext is not None:
                for x in ext:
                    if fname[-len(x):] == x:
                        fpaths.append(os.path.join(root, fname))
            else:
                fpaths.append(os.path.join(root, fname))
    return fpaths


def del_dir(dir_path):
    '''删除文件夹及其所有内容'''
    shutil.rmtree(dir_path)


def pickleFile(data, file):
    '''以二进制格式保存数据data到文件file'''
    with open(file, 'wb') as dbFile:
        pickle.dump(data, dbFile)


def unpickleFile(file):
    '''读取二进制格式文件file'''
    with open(file, 'rb') as dbFile:
        return pickle.load(dbFile)


def write_txt(lines, file, mode='w', **kwargs):
    '''
    将lines写入txt文件，文件路径为file
    lines为列表，每个元素为一行文本内容，末尾不包括换行符
    mode为写入模式，如'w'或'a'
    '''
    lines = [line + '\n' for line in lines]
    f = open(file, mode=mode, **kwargs)
    f.writelines(lines)
    f.close()


def read_lines(fpath, logger=None):
    '''读取txt文件中的所有行'''
    try:
        with open(fpath, 'r') as f:
            lines = f.readlines()
    except:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            try:
                with open(fpath, 'r', encoding='gbk') as f:
                    lines = f.readlines()
            except:
                logger_show('未正确识别文件编码格式，以二进制读取: {}'.format(fpath),
                            logger, 'warn')
                with open(fpath, 'rb') as f:
                    lines = f.readlines()
    return lines


def load_text(fpath, sep=',', del_first_line=False, del_first_col=False,
              to_pd=True, keep_header=True, encoding=None, del_last_col=False,
              logger=None):
    '''
    读取文本文件数据，要求文件每行一个样本

    Parameters
    ----------
    fpath: 文本文件路径
    sep: 字段分隔符，默认`,`
    del_first_line: 是否删除首行，默认不删除
    del_first_col: 是否删除首列，默认不删除
    to_pd: 是否输出为pandas.DataFrame，默认是
    keep_header: 输出为pandas.DataFrame时是否以首行作为列名，默认是
    encoding: 指定编码方式，默认不指定，不指定时会尝试以uft-8和gbk编码读取
    del_last_col: 是否删除最后一列，默认否
    logger: 日志记录器

    注：若del_first_line为True，则输出pandas.DataFrame没有列名

    Returns
    -------
    data: list或pandas.DataFrame
    '''

    if logger is None:
        logger = simple_logger()

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
    for line in lines:
        line = str(line)
        line = line.strip()
        line = line.split(sep)
        if del_first_col:
            line = line[1:]
        if del_last_col:
            line = line[:-1]
        data.append(line)

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

    if logger is None:
        logger = simple_logger()

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


def load_csv(fpath, del_unname_cols=True, logger=None, encoding=None,
             **kwargs):
    '''
    用pandas读取csv数据

    Args:
        fpath: csv文件路径
        del_unname_cols: 是否删除未命名列，默认删除
        logger: 日志记录器
        encoding: 指定编码方式，默认不指定，不指定时会尝试以uft-8和gbk编码读取
        **kwargs: 其它pd.read_csv支持的参数

    Returns:
        data: pandas.DataFrame
    '''

    if not os.path.exists(fpath):
        logger_show('文件不存在，返回None：{}!'.format(fpath),
                    logger, 'warn')
        return None

    try:
        data = pd.read_csv(fpath, encoding=encoding, **kwargs)
    except:
        try:
            data = pd.read_csv(fpath, encoding='utf-8', **kwargs)
        except:
            try:
                data = pd.read_csv(fpath, encoding='gbk', **kwargs)
            except:
                data = pd.read_csv(fpath, **kwargs)

    if del_unname_cols:
        del_cols = [x for x in data.columns if 'Unnamed:' in str(x)]
        if len(del_cols) > 0:
            data.drop(del_cols, axis=1, inplace=True)

    return data


def load_csvs(fdir, sort_cols=None, drop_duplicates=True,
              **kwargs_loadcsv):
    '''
    读取fdir中所有的csv文件，整合到一个df里面
    根据sort_cols指定列排序和去重
    '''
    files = os.listdir(fdir)
    files = [os.path.join(fdir, x) for x in files if x[-4:] == '.csv']
    data = []
    for file in files:
        df = load_csv(file, **kwargs_loadcsv)
        data.append(df)
    data = pd.concat(data, axis=0)
    if not isnull(sort_cols):
        data.sort_values(sort_cols, inplace=True)
        if drop_duplicates:
            dupcols = [sort_cols] if isinstance(sort_cols, str) else sort_cols
            data.drop_duplicates(subset=dupcols, inplace=True)
    return data


def load_excels(fdir, sort_cols=None, drop_duplicates=True,
                **kwargs_loadexcel):
    '''
    读取fdir中所有的excel文件，整合到一个df里面
    根据sort_cols指定列排序和去重
    '''
    files = os.listdir(fdir)
    files = [os.path.join(fdir, x) for x in files \
                         if x[-4:] == '.xls' or x[-5:] == '.xlsx']
    data = []
    for file in files:
        df = pd.read_excel(file, **kwargs_loadexcel)
        data.append(df)
    data = pd.concat(data, axis=0)
    if not isnull(sort_cols):
        data.sort_values(sort_cols, inplace=True)
        if drop_duplicates:
            dupcols = [sort_cols] if isinstance(sort_cols, str) else sort_cols
            data.drop_duplicates(subset=dupcols, inplace=True)
    return data


if __name__ == '__main__':
    fpath = './test/load_text_test_utf8.csv'
    data1 = load_text(fpath, encoding='gbk')
    data2 = load_csv(fpath, encoding='gbk')
