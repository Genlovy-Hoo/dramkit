# -*- coding: utf-8 -*-

import os
import json
import pickle
import shutil
import zipfile
import subprocess
import pandas as pd
# from .gentools import isnull
# from .logtools.utils_logger import logger_show
from dramkit.gentools import isnull
from dramkit.logtools.utils_logger import logger_show


def pickle_file(data, file):
    '''
    以二进制格式保存数据data到文件file

    Parameters
    ----------
    data :
        待保存内容
    file : str
        保存路径
    '''
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def unpickle_file(file):
    '''
    读取二进制格式文件file

    Parameters
    ----------
    file : str
        待读取文件路径
    '''
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_json(fpath, encoding=None, logger=None):
    '''
    读取json格式文件

    Parameters
    ----------
    fpath : str
        待读取文件路径
    encoding : str, None
        文件编码格式，若不指定，则尝试用utf-8和gbk编码读取
    logger : logging.Logger
        日志记录器


    :returns: `dict` - 返回读取数据
    '''

    if not os.path.exists(fpath):
        logger_show('文件不存在，返回None：{}'.format(fpath),
                    logger, 'warn')
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
                logger_show('读取%s出错，请检查文件（如编码或文件末尾多余字符等问题）！'%fpath,
                            logger, 'error')
                raise

    return data_json


def write_json(data, fpath, encoding=None):
    '''
    把data写入json格式文件

    Parameters
    ----------
    data : dict
        待写入数据
    fpath : str
        文件保存路径
    encoding : str, None
        文件编码格式
    '''
    with open(fpath, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_lines(fpath, encoding=None, logger=None):
    '''
    读取文本文件中的所有行

    Parameters
    ----------
    fpath : str
        待读取文件路径
    encoding : str, None
        文件编码格式，若不指定，则尝试用utf-8和gbk编码读取
    logger : None, logging.Logger
        日志记录器


    :returns: `list` - 文本文件中每行内容列表
    '''
    try:
        with open(fpath, 'r', encoding=encoding) as f:
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
                logger_show('未正确识别文件编码格式，以二进制读取: %s'%fpath,
                            logger, 'warn')
                with open(fpath, 'rb') as f:
                    lines = f.readlines()
    return lines


def write_txt(lines, file, mode='w', check_end=True, **kwargs):
    '''
    将列表lines中的内容写入文本文件中

    Parameters
    ----------
    lines : list
        列表，每个元素为一行文本内容
    file : str
        保存路径
    mode : str
        写入模式，如'w'或'a'
    check_end : bool
        是否检查每行行尾换行符，默认若行尾已经有换行符，则不再新添加换行符
    **kwargs :
        ``open`` 函数接受的关键字，如encoding等

    Note
    ----
    不同系统文本文件每行默认结尾字符不同:

        - linux下一般以`\\\\n`结尾
        - windows下一般以`\\\\r\\\\n`结尾
        - 苹果系统一般以`\\\\r`结尾
    '''
    if check_end:
        lines_ = []
        for x in lines:
            if not (x.endswith('\n') or x.endswith('\r\n') or x.endswith('\r')):
                x = x + '\n'
            lines_.append(x)
    else:
        lines_ = [x+'\n' for x in lines]
    f = open(file, mode=mode, **kwargs)
    f.writelines(lines_)
    f.close()


def load_text(fpath, sep=',', del_first_line=False, del_first_col=False,
              to_pd=True, keep_header=True, encoding=None, del_last_col=False,
              logger=None):
    '''
    读取文本文件数据，要求文件中每行存放一个数据样本

    Parameters
    ----------
    fpath : str
        文本文件路径
    sep : str
        字段分隔符，默认`,`
    del_first_line : bool
        是否删除首行，默认不删除

        .. note:: 若del_first_line为True，则输出pandas.DataFrame没有列名
    del_first_col : bool
        是否删除首列，默认不删除
    to_pd : bool
        是否输出为pandas.DataFrame，默认是
    keep_header : bool
        输出为pandas.DataFrame时是否以首行作为列名，默认是
    encoding : str, None
        指定编码方式，默认不指定时会尝试以uft-8和gbk编码读取
    del_last_col : bool
        是否删除最后一列，默认否
    logger : logging.Logger, None
        日志记录器


    :returns: `list, pandas.DataFrame` - 返回读取的数据
    '''

    if not os.path.exists(fpath):
        logger_show('文件不存在，返回None：%s'%fpath, logger, 'warn')
        return None

    lines = read_lines(fpath, encoding=encoding, logger=logger)

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


def load_csv(fpath, del_unname_cols=True, encoding=None,
             logger=None, **kwargs):
    '''
    用pandas读取csv数据

    Parameters
    ----------
    fpath : str
        csv文件路径
    del_unname_cols : bool
        是否删除未命名列，默认删除
    encoding : str, None
        指定编码方式，默认不指定，不指定时会尝试以uft-8和gbk编码读取
    logger : logging.Logger, None
        日志记录器
    **kwargs :
        其它 ``pd.read_csv`` 支持的参数


    :returns: `pandas.DataFrame` - 读取的数据
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
    读取指定文件夹中所有的csv文件，整合到一个df里面

    Parameters
    ----------
    fdir : str
        文件夹路径
    sort_cols : None, str, list
        根据sort_cols指定列排序(升序)和去重
    drop_duplicates : bool
        是否删除重复值(重复值保留第一条)
    **kwargs_loadcsv :
        :func:`dramkit.iotools.load_csv` 接受的其它参数


    :returns: `pandas.DataFrame` - 读取的数据
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
                **kwargs_readexcel):
    '''
    读取指定文件夹中所有的excel文件，整合到一个df里面

    Parameters
    ----------
    fdir : str
        文件夹路径
    sort_cols : None, str, list
        根据sort_cols指定列排序(升序)和去重
    drop_duplicates : bool
        是否删除重复值(重复值保留第一条)
    **kwargs_readexcel :
        ``pd.read_excel`` 接受的其它参数


    :returns: `pandas.DataFrame` - 读取的数据
    '''
    files = os.listdir(fdir)
    files = [os.path.join(fdir, x) for x in files if x[-4:] == '.xls' or x[-5:] == '.xlsx']
    data = []
    for file in files:
        df = pd.read_excel(file, **kwargs_readexcel)
        data.append(df)
    data = pd.concat(data, axis=0)
    if not isnull(sort_cols):
        data.sort_values(sort_cols, inplace=True)
        if drop_duplicates:
            dupcols = [sort_cols] if isinstance(sort_cols, str) else sort_cols
            data.drop_duplicates(subset=dupcols, inplace=True)
    return data


def get_all_files(dir_path, ext=None, include_dir=False, abspath=False):
    '''
	获取指定文件夹及其子文件夹中所有的文件路径

    Parameters
    ----------
    dir_path : str
        文件夹路径
    ext : None, str, list
        指定文件后缀列表，若为None，则包含所有类型文件
    include_dir : bool
        返回结果中是否包含文件夹路径，默认不包含（即只返回文件路径）
    abspath : bool
        是否返回绝对路径，默认返回相对路径


    :returns: `list` - 文件路径列表
	'''
    if not (ext is None or isinstance(ext, list) or isinstance(ext, str)):
        raise ValueError('`ext`必须为None或str或list类型！')
    if ext is not None and isinstance(ext, str):
        ext = [ext]
    fpaths = []
    for root, dirs, files in os.walk(dir_path):
        if include_dir:
            fpaths.append(root)
        for fname in files:
            if ext is not None:
                for x in ext:
                    if fname[-len(x):] == x:
                        fpaths.append(os.path.join(root, fname))
            else:
                fpaths.append(os.path.join(root, fname))
    if abspath:
        fpaths = [os.path.abspath(x) for x in fpaths]
    return fpaths


def zip_files(zip_path, fpaths, keep_ori_path=True, keep_zip_new=True):
    '''
    使用zipfile将指定路径列表(不包括子文件(夹)内容)打包为.zip文件

    Parameters
    ----------
    zip_path : str
        zip压缩包保存路径
    fpaths : list
        需要压缩的路径列表(应为相对路径)
    keep_ori_path : bool
        - 若为True, 则压缩文件会保留fpaths中文件的原始路径
        - 若为False, 则fpaths中所有文件在压缩文件中都在统一根目录下
    keep_zip_new : bool
        若为True，将覆盖已有压缩文件，否则在已有文件里面新增
    '''
    if keep_zip_new:
        f = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    else:
        f = zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED)
    if keep_ori_path:
        for fpath in fpaths:
            f.write(fpath)
    else:
        for fpath in fpaths:
            file = os.path.basename(fpath)
            f.write(fpath, file)
    f.close()
    
    
def zip_fpath(fpath, zip_path=None, **kwargs):
    '''
    使用zipfile压缩单个文件(夹)下所有内容为.zip文件

    Parameters
    ----------
    fpath : str
        待压缩文件(夹)路径(应为相对路径)
    zip_path : None, str
        压缩文件保存路径，若为None，则为fpath路径加后缀
    **kwargs :
        :func:`dramkit.iotools.zip_files` 接受的参数
    '''
    if isnull(zip_path):
        if os.path.isdir(fpath) and fpath[-1] == '/':
            zip_path = fpath[:-1] + '.zip'
        else:
            zip_path = fpath + '.zip'
    if os.path.isfile(fpath): # fpath为文件
        zip_files(zip_path, [fpath], **kwargs)
    elif os.path.isdir(fpath): # fpath为文件夹
        fpaths = get_all_files(fpath, include_dir=True)
        zip_files(zip_path, fpaths, **kwargs)


def zip_fpaths(zip_path, fpaths, **kwargs):
    '''
    使用zipfile将指定路径列表(包括子文件(夹)所有内容)打包为.zip文件

    Parameters
    ----------
    zip_path : str
        zip压缩包保存路径
    fpaths : list
        需要压缩的路径列表(可为文件也可为文件夹, 应为相对路径)
    **kwargs :
        :func:`dramkit.iotools.zip_files` 接受的参数
    '''
    all_paths = []
    for fpath in fpaths:
        if os.path.isfile(fpath):
            all_paths.append(fpath)
        elif os.path.isdir(fpath):
            all_paths += get_all_files(fpath, include_dir=True)
    zip_files(zip_path, all_paths, **kwargs)


def zip_extract():
    '''用zipfile解压文件，待实现'''
    raise NotImplementedError


def zip_fpath_7z(fpath, zip_path=None, mode='zip', pwd=None, keep_zip_new=True):
    '''
    7z命令压缩单个文件(夹)到.zip文件

    Parameters
    ----------
    fpath : str
        待压缩文件(夹)路径
    zip_path : None, str
        压缩文件保存路径，若为None，则为fpath路径加后缀
    mode : str
        压缩文件后缀，可选['7z', 'zip']
    pwd : str
        密码字符串
    keep_zip_new : bool
        - 若为True，则zip_path将覆盖原来已经存在的文件
        - 若为False，则zip_path将在原来已有的文件中新增需要压缩的文件
    '''

    fpath = os.path.abspath(fpath) # 绝对路径

    if isnull(zip_path):
        if os.path.isdir(fpath) and fpath[-1] == '/':
            zip_path = fpath[:-1] + '.zip'
        else:
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
    7z命令压缩多个文件(夹)列表到.zip文件

    Parameters
    ----------
    zip_path : str
        zip压缩包保存路径
    fpaths : list
        待压缩文件(夹)路径列表

        .. warning:: fpaths太长的时候可能会报错
    mode : str
        压缩文件后缀，可选['7z', 'zip']
    pwd : str
        密码字符串
    keep_zip_new : bool
        - 若为True，则zip_path将覆盖原来已经存在的文件
        - 若为False，则zip_path将在原来已有的文件中新增需要压缩的文件
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


def extract_7z():
    '''7z命令解压文件，待实现'''
    raise NotImplementedError


def cmdrun(cmd_str):
    '''调用cmd执行cmd_str命令，待实现'''
    raise NotImplementedError


def rename_files_in_dir(dir_path, func_rename):
    '''
    对指定文件夹中的文件进行批量重命名
    
    Parameters
    ----------
    dir_path : str
        目标文件夹
    func_rename : function
        命名规则函数: name_new = func_rename(name)
    '''
    files = os.listdir(dir_path)
    for name in files:
        name_new = func_rename(name)
        old_path = os.path.join(dir_path, name)
        new_path = os.path.join(dir_path, name_new)
        os.rename(old_path, new_path)


def del_dir(dir_path):
    '''删除dir_path指定文件夹及其所有内容'''
    shutil.rmtree(dir_path)
    

def copy_file():
	'''复制文件，待实现'''
	raise NotImplementedError


def copy_dir():
    '''复制文件夹，待实现'''
    raise NotImplementedError


def find_files_include_str(target_str, root_dir=None,
                           file_types='default', logger=None):
    '''
    在指定目录下的文件中，查找那些文件里面包含了目标字符串

    Parameters
    ----------
    target_str : str
        目标字符串
    root_dir : str, None
        目标文件夹，目标字符串在此文件夹及其所有子文件内所有文本文件中搜索
        若为None，则在os.getcwd()下搜索
    file_types : None, str, list
        指定查找的文件后缀范围:

        - None, 在所有文件中查找
        - str, 指定一类文件后缀, 如'.py'表示在Python脚本中查找
        - list, 指定一个后缀列表, 如['.py', '.txt']
    logger : None, logging.Logger
        日志记录器


    :returns: `dict` - 返回dict, key为找到的文件路径，value为包含目标字符串的文本内容(仅第一次出现的位置)
    '''
    if root_dir is None:
        root_dir = os.getcwd()
    all_files = get_all_files(root_dir, ext=file_types)
    files = []
    for fpath in all_files:
        lines = read_lines(fpath, logger=logger)
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
    fpath = './test/load_text_test_utf8.csv'
    data1 = load_text(fpath, encoding='gbk')
    data2 = load_csv(fpath, encoding='gbk')
