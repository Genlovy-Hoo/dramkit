# -*- coding: utf-8 -*-

import time
from copy import copy
import numpy as np
import pandas as pd
import cx_Oracle as cxorcl
from dramkit.gentools import (isnull,
                              check_list_arg,
                              df_na2value,
                              change_dict_key,
                              list_eq)


class CxOracle(object):
    
    def __init__(self,
                 host='localhost', user='test',
                 password='xxxxxxxxxxx', database='orclpdb',
                 port=1521, logger=None, **kwargs):
        self.__db_conn_args = {'host': host,
                               'user': user,
                               'password': password,
                               'database': database,
                               'port': port,
                               }
        self.__db_conn_args.update(kwargs)
        self.conn = get_conn(host=host,
                             user=user,
                             password=password,
                             database=database,
                             port=port,
                             **kwargs)
        self.db_name = database
        self.logger = logger
        
    def copy(self):
        res = copy(self)
        res.conn = get_conn(**self.__db_conn_args)
        return res
        
    def execute_sql(self, sql_str, to_df=True):
        res = execute_sql(conn=self.conn, sql_str=sql_str,
                          to_df=to_df)
        return res
    
    def get_tables(self):
        return get_tables(self.conn)
    
    def get_fields(self, tb_name):
        return get_fields(self.conn, tb_name)
    
    def has_table(self, tb_name):
        return has_table(self.conn, tb_name)
    
    def clear_data(self, tb_name):
        return clear_data(self.conn, tb_name)
    
    def drop_table(self, tb_name, purge=True):
        return drop_table(self.conn, tb_name, purge=purge)
    
    def get_create_table_sql(self, tb_name):
        return get_create_table_sql(self.conn, tb_name)
    
    def create_table(self, tb_name, cols_info,
                     idcols=None, force=False):
        return create_table(self.conn, tb_name, cols_info,
                            idcols=idcols, force=force)
        
    def df_to_sql(self, df, tb_name, act_type='insert',
                  cols=None, idcols=None,
                  col_types={}, na_val=None,
                  inf_val='inf', _inf_val='-inf',
                  **kwargs_cols):
        df_to_sql(df, self.conn, tb_name,
                  act_type=act_type, cols=cols,
                  idcols=idcols, col_types=col_types,
                  na_val=na_val, inf_val=inf_val,
                  _inf_val=_inf_val, **kwargs_cols)


def get_conn(host='localhost', user='test',
             password=None, database='orclpdb',
             port=1521, **kwargs):
    '''
    连接数据库
    
    Examples
    --------
    >>> host, user = 'localhost', 'test',
    >>> database, port = 'orclpdb', 1521
    >>> password = 'xxxxxxxxxxx'
    >>> con1 = get_conn(host=host, user=user,
    ...                 password=password,
    ...                 database=database, port=port)
    >>> host, user = 'localhost', 'c##test',
    >>> database, port = 'orcl', 1521
    >>> password = 'xxxxxxxxxxx'
    >>> con2 = get_conn(host=host, user=user,
    ...                 password=password,
    ...                 database=database, port=port)
    '''
    url_ = '{}/{}@{}:{}/{}'.format(
           user, password, host, port, database)
    conn = cxorcl.connect(url_, **kwargs)
    return conn


def _get_test_conn():
    return get_conn(host='localhost', user='test',
                    password='xxxxxxxxxxx',
                    database='orclpdb', port=1521)


def execute_sql(conn, sql_str, to_df=True):
    '''
    执行sql语句并返回结果
    
    Examples
    --------
    >>> conn = _get_test_conn()
    >>> df1 = execute_sql(conn, 'select * from test1')
    >>> # execute_sql(conn, 'drop table test2 purge')
    >>> execute_sql(conn,
    ...             """create table test2
    ...             (code varchar(20), year int, val1 float)""")
    >>> df2 = execute_sql(conn, 'select * from test2')
    >>> execute_sql(conn,
                    """insert into test2 values 
                    ('a', 2022, 1)""")
    >>> df2 = execute_sql(conn, 'select * from test2')
    '''
    cur = conn.cursor()
    cur.execute(sql_str)
    if isnull(cur.description):
        cur.close()
        conn.commit()
        return None
    res = cur.fetchall()
    if to_df:
        cols = [x[0] for x in cur.description]
        res = pd.DataFrame(res, columns=cols)
    cur.close()
    conn.commit()
    return res


def get_fields(conn, tb_name):
    '''获取表字段名列表'''
    sql_str = '''SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
                 FROM user_tab_columns
                 WHERE table_name = UPPER('{}')
              '''.format(tb_name)
    fields_info = execute_sql(conn, sql_str, to_df=True)
    fields = fields_info['COLUMN_NAME'].tolist()
    return fields, fields_info


def clear_data(conn, tb_name):
    '''清空表中数据'''
    sql = 'TRUNCATE TABLE {}'.format(tb_name)
    # sql = 'DELETE FROM {}'.format(tb_name)
    execute_sql(conn, sql, to_df=False)


def has_table(conn, tb_name):
    sql = "SELECT COUNT(*) n FROM user_tables WHERE table_name=UPPER('{}')".format(tb_name)
    n = execute_sql(conn, sql, to_df=False)[0][0]
    if n > 0:
        return True
    return False


def copy_table(conn, tb_name, with_data=False):
    '''复制表'''
    raise NotImplementedError


def create_table(conn, tb_name, cols_info, idcols=None,
                 force=False):
    '''
    新建表
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> idcols = None
    >>> idcols = ['a', 'c']
    >>> create_table(conn, 'test2',
    ...              ('a VARCHAR2(255)',
    ...               'b FLOAT DEFAULT 1 NOT NULL',
    ...               'c DATE'),
    ...              idcols=idcols, force=True)
    '''
    has_ = has_table(conn, tb_name)
    if has_ and not force:
        return
    if has_ and force:
        execute_sql(conn, 'DROP TABLE %s PURGE'%tb_name)
    idcols = check_list_arg(idcols, allow_none=True)
    colstr = '('
    colstr = colstr + ', '.join(cols_info)
    if not isnull(idcols):
        pkstr = '\nCONSTRAINT PK_{} PRIMARY KEY ({})'.format(tb_name, ', '.join(idcols))
        colstr = colstr + ',' + pkstr + ')'
    else:
        colstr = colstr + ')'
    sql = '''CREATE TABLE {}
             {}
          '''.format(tb_name, colstr)
    execute_sql(conn, sql)
    
    
def get_create_table_sql(conn, tb_name):
    '''查询表创建语句'''
    sql_str = "SELECT dbms_metadata.get_ddl('TABLE', '{}') FROM dual".format(tb_name.upper())
    res = execute_sql(conn, sql_str, to_df=False)[0][0]
    return str(res)


def drop_table(conn, tb_name, purge=True):
    '''删除表'''
    if has_table(conn, tb_name):
        if purge:
            execute_sql(conn, 'DROP TABLE %s PURGE'%tb_name)
        else:
            execute_sql(conn, 'DROP TABLE %s'%tb_name)
            
            
def get_tables(conn):
    df = execute_sql(conn, 'SELECT * FROM user_tables')
    tbs = df['TABLE_NAME'].tolist()
    return tbs, df


def get_cols_info_df(df, cols=None, col_types={}, 
                     all2str=False, big_text_cols=[]):
    '''
    根据pd.DataFrame中的列cols，识别对应列在Oracle中的字段类别
    
    Parameters
    ----------
    df : pandas.DataFrame
        待识别数据
    cols : list, None
        待识别列名列表，默认所有列
    col_types : dict
        指定列类型，如{'col1': 'VARCHAR2(20)', 'col2': 'NUMBER(10,0)'}，指定的列不做判断，直接返回
    all2str : bool
        若为True，则数据均按文本类型处理
    big_text_cols : str, list
        文本是否为长文本，若为'all'，则全部按长文本处理，若为list，则list指定列按长文本处理
        
    Returns
    -------
    cols_info : str
        列类型信息，格式如'col1 col1_type, col2 col2_type, ...'
    dtype : dict
        字典格式的类类型信息
    placeholder : str
        占位符信息，格式如':1, :2, ...'
    
    References
    ----------
    https://blog.csdn.net/tonydz0523/article/details/82529941
    '''

    if not cols is None:
        df = df.reindex(columns=cols)
    cols = df.columns.tolist()
    
    assert big_text_cols == 'all' or \
            isinstance(big_text_cols, list)
    if big_text_cols == 'all':
        big_text_cols = cols

    types = df.dtypes
    cols_info, placeholder, cols_info_dict = [], [], {}
    for k in range(0, len(cols)):
        col = cols[k]
        if col in col_types:
            char = col + ' ' + col_types[col]
        elif all2str:
            if col in big_text_cols:
                char = col + ' CLOB'
            else:
                char = col + ' VARCHAR2(255)'
        elif 'int' in str(types[col]):
            char = col + ' INT' # NUMBER(*,0)
        elif 'float' in str(types[col]):
            char = col + ' FLOAT'
        elif 'object' in str(types[col]):
            if col in big_text_cols:
                char = col + ' CLOB'
            else:
                char = col + ' VARCHAR2(255)'
        elif 'datetime' in str(types[col]):
            char = col + ' DATE'
        else:
            raise ValueError('未识别（暂不支持）的字段类型: %s！'%col)
        char_ = ':%s'%(k+1)
        cols_info.append(char)
        placeholder.append(char_)
        cols_info_dict[col] = (char, char_)
    cols_info, placeholder = ', '.join(cols_info), ', '.join(placeholder)
    dtype = {k: v[0].split(' ')[-1] for k, v in cols_info_dict.items()}
    
    return cols_info, dtype, placeholder, cols_info_dict


def _get_tmp_tb_name(conn, tmp_tb_name):
    assert isinstance(tmp_tb_name, str)
    tmp_tb_name = tmp_tb_name.upper()
    tbs, _ = get_tables(conn)
    tbs = [x.upper() for x in tbs]
    while tmp_tb_name in tbs:
        tmp_tb_name += '_'
    return tmp_tb_name


def merge_into(conn, tb_tgt, tb_src, cols, idcols, rep_keep='src'):
    assert rep_keep in ['src', 'tgt']
    assert isinstance(cols, list) and isinstance(idcols, list)
    noidcols = [x for x in cols if not x in idcols]
    on_ = ' AND '.join(['%s.%s=%s.%s'%(tb_tgt, c, tb_src, c) for c in idcols])
    update_ = ''
    if rep_keep == 'src':
        update_ = '''WHEN MATCHED THEN 
                     UPDATE SET {} 
                  '''.format(', '.join(['%s=%s.%s'%(c, tb_src, c) for c in noidcols]))
    insert_ = 'INSERT ({}) VALUES ({})'.format(', '.join(cols), ', '.join(['%s.%s'%(tb_src, c) for c in cols]))
    sql = '''MERGE INTO {} 
             USING {} 
             ON ({}) 
             {}
             WHEN NOT MATCHED THEN
             {}
          '''.format(tb_tgt, tb_src, on_, update_, insert_)
    execute_sql(conn, sql)


def df_to_sql(df, conn, tb_name, act_type='insert',
              cols=None, idcols=None, col_types={},
              na_val=None, inf_val='inf', _inf_val='-inf',
              **kwargs_cols):
    '''
    把pandas.DataFrame存入Oracle数据库中
    
    Parameters
    ----------
    df : pandas.DataFrame
        待存数据
    conn : cx_oracle.connect
        cx_oracle.connect数据库连接对象
    tb_name : str
        存入的表名
    act_type : str
        | 存入方式：
        |     若为'ignore_tb'，则当表已经存在时不进行任何操作
        |     若为'new'，则新建表（若原表已存在，则会先删除再重建）
        |     若为'clear'，则先清空已存在数据（若原表已存在）
        |     若为'insert'，则直接插入
        |     若为'replace'，则将已存在的数据更新，不存在的行和列都新插入
        |     若为'insert_ignore'，则已有的行不更新，不存在的行新插入
        |     若为'insert_newcols'（谨慎使用），则对已存在的列同直接insert（但是遇到已存在的行时不会报错，可能会丢失数据），有新列则插入新列内容
        |     若为'insert_ignore_newcols'，则对已存在的列同直接insert_ignore，有新列则插入新列内容
    cols : None, list
        需要存入的数据列
    db_name : str
        存入的数据库名
    na_val : None, bool, str
        设置df中na值的填充值
    idcols : None, str, list
        设置唯一值标识字段，只有在表不存在需要新建的时候才起作用
    col_types : dict
        指定字段类型
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> tb_name = 'test1'
    >>> # idcols = None
    >>> idcols = ['code', 'year']
    >>> df0 = pd.DataFrame({'code': ['001', '002', '003', '004', '005', '006'],
    ...                     'year': ['2011', '2012', '2013', '2014', '2015', '2016'],
    ...                     'value': [1, 2, 3, 4, 5, 6],
    ...                     'value1': [1, 2, '3', 4, 5, 6],
    ...                     'value0': ['1a', '2b', '3c', '4d', '5e', '6f']})
    >>> df = pd.DataFrame({'code': ['001', '002', '003', '004', '005', '006'],
    ...                    'year': ['2011', '2012', '2013', '2014', '2015', '2016'],
    ...                    'value': [1, 2, 3, np.nan, np.inf, -np.inf],
    ...                    'value0': ['1a', '2b', '3c', '4d', '5e', '6f']})
    >>> df1 = pd.DataFrame({'code': ['006', '007', '008', '009'],
    ...                     'year': ['2016', '2017', '2018', '2019'],
    ...                     'value': [66, 7, 8, 9],
    ...                     'VALUE2': [10, 11, 12, 13],
    ...                     'VALUE3': ['10a', '11b', 12, '13']})
    >>> df_to_sql(df, conn, tb_name, act_type='new', idcols=idcols)
    >>> df_to_sql(df1, conn, tb_name, act_type='replace', idcols=idcols)
    '''
    
    assert act_type in ['ignore_tb', 'new', 'clear', 'insert', 'insert_ignore', 'replace',
                        'insert_newcols', 'insert_ignore_newcols']
    if act_type == 'ignore_tb' and has_table(conn, tb_name):
        return
    
    # 待入库字段检查
    cols = check_list_arg(cols, allow_none=True)
    # 字段名称统一处理为大写
    df = df.rename(columns={x: x.upper() for x in df.columns})
    if not isnull(cols):
        cols = [x.upper() for x in cols]
        df = df[cols].copy()
    else:
        cols = df.columns.tolist()
       
    # inf和na值处理，须先处理inf再处理na，否则可能报错
    if inf_val != False:
        df = df.replace({np.inf: inf_val})
    if _inf_val != False:
        df = df.replace({-np.inf: _inf_val})
    if na_val != False:
        df = df_na2value(df, value=na_val)
        
    # object统一处理为字符串型
    for col in df.columns:
        if str(df[col].dtype) == 'object':
            df[col] = df[col].astype(str)

    cur_ = conn.cursor()

    cols_info, dtype, ph, cols_info_dict = get_cols_info_df(
        df, cols=cols, col_types=change_dict_key(col_types, lambda x: x.upper()),
        **kwargs_cols)

    # 清空表
    if act_type == 'clear':
        if has_table(conn, tb_name):
            clear_data(conn, tb_name)
    # 表不存在则新建
    idcols = check_list_arg(idcols, allow_none=True)
    idcols = [x.upper() for x in idcols] if not isnull(idcols) else idcols
    force = True if act_type == 'new' else False
    create_table(conn, tb_name, cols_info.split(', '),
                 idcols=idcols, force=force)
    
    # 表字段列表
    fields = get_fields(conn, tb_name)[0]
    fields = [x.upper() for x in fields]

    # 检查字段是否已经存在，不存在新建
    # TODO: 不存在的字段中有idcols中的字段的情况处理
    # （可以先新建字段再重新设置联合唯一索引，不过Oracle貌似不用处理，待确定）
    cols_loss = [x for x in cols if x not in fields]
    for col in cols_loss:
        cur_.execute('ALTER TABLE {} ADD {}'.format(
             tb_name, cols_info_dict[col][0]))
    
    # 唯一值字段
    cur_.execute("SELECT COLUMN_NAME FROM user_ind_columns WHERE TABLE_NAME = '%s'"%tb_name.upper())
    idcols_ = list(set([x[0].upper() for x in cur_.fetchall()]))    
    if any([not x in cols for x in idcols_]):
        raise ValueError('待存入数据中必须包含所有唯一值字段！')
        
    # TODO: idcols和idcols_即已有唯一索引字段不符合的情况处理
    assert list_eq(idcols, idcols_, order=False), '`idcols`必须和表唯一值索引相同'
    
    # 数据更新
    values = df.values.tolist()
    if isnull(idcols) or act_type in ['new', 'clear']:        
        cur_.executemany('INSERT INTO {a} ({b}) VALUES ({c})'.format(
                             a=tb_name, b=','.join(cols), c=ph),
                         values)
    elif act_type == 'insert':
        # 批量插入新数据
        try:
            cur_.executemany('INSERT INTO {a} ({b}) VALUES ({c})'.format(
                                 a=tb_name, b=','.join(cols), c=ph),
                             values)
        except:
            for col in cols_loss:
                cur_.execute('ALTER TABLE {} DROP COLUMN {}'.format(tb_name, col))
            raise
    elif act_type == 'insert_ignore':
        cur_.executemany('''INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX({tb}({ids})) */
                            INTO {a} ({b}) VALUES ({c})
                         '''.format(tb=tb_name, ids=', '.join(idcols),
                                    a=tb_name, b=','.join(cols), c=ph),
                         values)
    else:
        tb_name_tmp = _get_tmp_tb_name(conn, tb_name+'_TMP')
        df_to_sql(df, conn, tb_name_tmp, act_type='new',
                  cols=cols, idcols=idcols,
                  col_types={k: v[0].split(' ')[1] for k, v in cols_info_dict.items()},
                  na_val=na_val, inf_val=inf_val,
                  _inf_val=_inf_val, **kwargs_cols)
        try:
            # time.sleep(2)
            if act_type == 'replace':
                # 有则更新，无则插入
                merge_into(conn, tb_name, tb_name_tmp, cols, idcols,
                           rep_keep='src')
            # elif act_type == 'insert_ignore':
            #     # 若有重复，保留已存在的不更新
            #     merge_into(conn, tb_name, tb_name_tmp, cols, idcols,
            #                rep_keep='tgt')
            else:
                # 先处理已存在字段
                oldcols = [x for x in cols if x in fields]
                oldcols = list(set(idcols + oldcols))
                if len(oldcols) > 0:
                    values = df[oldcols].values.tolist()
                    if act_type == 'insert_newcols':
                        try:
                            cur_.executemany('INSERT INTO {a} ({b}) VALUES ({c})'.format(
                                                 a=tb_name, b=','.join(oldcols),
                                                 c=', '.join([':%s'%x for x in range(1, len(oldcols)+1)])),
                                             values)
                        except:
                            pass
                    else:
                        merge_into(conn, tb_name, tb_name_tmp, oldcols, idcols,
                                   rep_keep='tgt')                
                # 再处理新增字段
                if len(cols_loss) > 0:
                    newcols = list(set(idcols + cols_loss))
                    merge_into(conn, tb_name, tb_name_tmp, newcols, idcols,
                               rep_keep='src')
        finally:
            drop_table(conn, tb_name_tmp, purge=True)
    cur_.close()
    conn.commit()
    
