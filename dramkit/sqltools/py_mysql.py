# -*- coding: utf-8 -*-

import pymysql
import pandas as pd
from dramkit.gentools import isnull


class PyMySQL(object):
    
    def __init__(self,
                 host='localhost', user='root',
                 password=None, database=None,
                 port=3306, **kwargs):
        self.conn = get_conn(host=host,
                             user=user,
                             password=password,
                             database=database,
                             port=port,
                             **kwargs)
        self.db_name_ori = database
        self.db_name = database
    
    def select_db(self, db_name):
        if not isnull(db_name):
            self.conn.select_db(db_name)
            self.db_name = db_name
        
    def reset_db_ori(self):
        self.select_db(self.db_name_ori)
        
    def reset_db(self, db_name):
        if db_name != self.db_name:
            self.select_db(self.db_name)
        
    def execute_sql(self, sql_str, db_name=None, to_df=True):
        res = execute_sql(conn=self.conn, sql_str=sql_str,
                          db_name=db_name, to_df=to_df)
        self.reset_db(db_name)
        return res
    
    def _check_db(self, db_name):
        if isnull(db_name):
            return self.db_name
        return db_name
    
    def get_primary_keys(self, tb_name, db_name=None):
        db_name = self._check_db(db_name)
        res = get_primary_keys(conn=self.conn,
                               tb_name=tb_name,
                               db_name=db_name)
        self.reset_db(db_name)
        return res
    
    def get_uni_indexs(self, tb_name, db_name=None):
        res = get_uni_indexs(conn=self.conn,
                             tb_name=tb_name,
                             db_name=db_name)
        self.reset_db(db_name)
        return res
    
    def set_primary_key(self, tb_name, cols_key, db_name=None):
        db_name = self._check_db(db_name)
        set_primary_key(conn=self.conn, tb_name=tb_name,
                        cols_key=cols_key, db_name=db_name)
        self.reset_db(db_name)
    
    def set_uni_index(self, tb_name, cols_uni,
                      index_name=None, db_name=None):
        set_uni_index(conn=self.conn, tb_name=tb_name,
                      cols_uni=cols_uni, db_name=db_name,
                      index_name=index_name)
        self.reset_db(db_name)
    
    def df_to_mysql(self, df, tb_name, act_type='insert',
                    cols=None, db_name=None):
        db_name = self._check_db(db_name)
        df_to_mysql(df, conn=self.conn, tb_name=tb_name,
                    act_type=act_type, db_name=db_name,
                    cols=cols)
        self.reset_db(db_name)


def get_conn(host='localhost', user='root', password=None,
             database=None, port=3306, **kwargs):
    '''连接数据库'''
    conn = pymysql.connect(host=host,
                           user=user,
                           passwd=password,
                           database=database,
                           port=port,
                           **kwargs)
    return conn


def execute_sql(conn, sql_str, db_name=None, to_df=True):
    '''执行sql语句并返回结果'''
    cur = conn.cursor()
    if not isnull(db_name):
        cur.execute('USE {};'.format(db_name))
    cur.execute('SELECT DATABASE();')
    cur.execute(sql_str)
    res = cur.fetchall()
    if isnull(cur.description):
        return None
    if to_df:
        cols = [x[0] for x in cur.description]
        res = pd.DataFrame(res, columns=cols)
    cur.close()
    conn.commit()
    return res


def get_primary_keys(conn, tb_name, db_name=None):
    '''获取主键列名'''
    sql = '''SELECT column_name
             FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
             WHERE table_name='{}' 
             AND CONSTRAINT_SCHEMA='{}'
             AND constraint_name='PRIMARY';
          '''.format(tb_name, db_name)
    keys = execute_sql(conn, sql, db_name=None, to_df=False)
    keys = [x[0] for x in keys]
    return keys


def get_uni_indexs(conn, tb_name, db_name=None):
    '''获取唯一索引列名'''
    sql = '''SHOW INDEX FROM {}
             WHERE Non_unique = 0
             AND Key_name <> 'PRIMARY';
          '''.format(tb_name)
    indexs = execute_sql(conn, sql, db_name=db_name, to_df=False)
    indexs = [x[4] for x in indexs]
    return list(set(indexs))


def set_primary_key(conn, tb_name, cols_key, db_name=None):
    '''主键设置'''
    assert isinstance(cols_key, (str, list, tuple))
    if isinstance(cols_key, str):
        cols_key = [cols_key]
    keys = get_primary_keys(conn, tb_name, db_name=db_name)
    cur = conn.cursor()
    if not isnull(db_name):
        cur.execute('USE {};'.format(db_name))
    if any([not x in keys for x in cols_key]):
        if len(keys) > 0:
            # 先删除主键再设置主键
            cur.execute('ALTER TABLE {} DROP PRIMARY KEY;'.format(tb_name))
        cur.execute('ALTER TABLE {} ADD PRIMARY KEY({});'.format(
            tb_name, ','.join(cols_key)))
    cur.close()
    conn.commit()
    

def set_uni_index(conn, tb_name, cols_uni,
                  db_name=None, index_name=None):
    '''唯一值索引设置'''
    assert isinstance(cols_uni, (str, list, tuple))
    if isinstance(cols_uni, str):
        cols_uni = [cols_uni]
    cur = conn.cursor()
    if not isnull(db_name):
        cur.execute('USE {};'.format(db_name))
    keys = get_uni_indexs(conn, tb_name, db_name=db_name)
    if any([not x in keys for x in cols_uni]):
        if isnull(index_name):
            cur.execute('ALTER TABLE {} ADD UNIQUE INDEX({});'.format(
                tb_name, ','.join(cols_uni)))
        else:
            try:
                # 先尝试删除已有同名index
                cur.execute('DROP INDEX {} ON {};'.format(
                                            index_name, tb_name))
            except:
                pass
            cur.execute('ALTER TABLE {} ADD UNIQUE INDEX {} ({});'.format(
                tb_name, index_name, ','.join(cols_uni)))
    cur.close()
    conn.commit()


def _get_cols_info_df(df, cols=None):
    '''
    | 根据pd.DataFrame中的列cols，识别对应列在MySQL中的字段类别
    
    Parameters
    ----------
    df : pandas.DataFrame
        待识别数据
    cols : list, None
        待识别列名列表，默认所有列
        
    Returns
    -------
    cols_info : str
        列类型信息，格式如'col1 col1_type, col2 col2_type, ...'
    cols_type : str
        列格式信息，格式如'%s, %s, ...'
    
    References
    ----------
    https://blog.csdn.net/tonydz0523/article/details/82529941
    '''

    if not cols is None:
        df = df.reindex(columns=cols)
    cols = df.columns.tolist()

    types = df.dtypes
    cols_info, cols_type, cols_info_dict = [], [], {}
    for col in cols:
        if 'int' in str(types[col]):
            char = col + ' INT'
            char_ = '%s'
        elif 'float' in str(types[col]):
            char = col + ' FLOAT'
            char_ = '%s'
        elif 'object' in str(types[col]):
            char = col + ' VARCHAR(255)'
            char_ = '%s'
        elif 'datetime' in str(types[col]):
            char = col + ' DATETIME'
            char_ = '%s'
        cols_info.append(char)
        cols_type.append(char_)
        cols_info_dict[col] = (char, char_)
    cols_info, cols_type = ','.join(cols_info), ','.join(cols_type)

    return cols_info, cols_type, cols_info_dict


def df_to_mysql(df, conn, tb_name, act_type='insert',
                db_name=None, cols=None):
    '''
    把pandas.DataFrame存入MySQL数据库中
    
    Parameters
    ----------
    df : pandas.DataFrame
        待存数据
    conn : pymysql.connect
        pymysql.connect数据库连接对象
    db_name : str
        存入的数据库名
    tb_name : str
        存入的表名
    act_type : str
        | 存入方式：
        |     若为'new'，则新建表（若原表已存在，则会删除重建）
        |     若为'insert'，则直接插入
        |     若为'replace'，则将已存在的数据更新，不存在的新插入
        |     若为'insert_ignore'，则已有的不更新，不存在的新插入
    cols : None, list
        需要存入的数据列
    
    Examples
    --------
    >>> db_name = 'test'
    >>> tb_name = 'test1'
    >>> df = pd.DataFrame({'code': ['001', '002', '003'],
    ...                    'year': ['2011', '2012', '2013'],
    ...                    'value': [1, 2, 3]})
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> df_to_mysql(df, conn, tb_name, 'new', db_name)
    >>> df2 = pd.DataFrame({'code': ['001', '002', '003'],
    ...                     'year': ['2011', '2012', '2014'],
    ...                     'value': [1, 2, 4],
    ...                     'value2': [2, 3, 4]})
    >>> df_to_mysql(df2, conn, tb_name, 'insert', db_name,
                    cols='value2')
    >>> df_to_mysql(df, conn, tb_name, 'new', db_name)
    >>> set_primary_key(conn, tb_name, 'code', db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'replace', db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'insert', db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'insert_ignore', db_name)
    
    References
    -----------    
    - https://blog.csdn.net/tonydz0523/article/details/82529941
    - https://blog.csdn.net/weixin_44848356/article/details/119113174
    - https://blog.csdn.net/weixin_42272869/article/details/116480732
    '''
    
    assert act_type in ['new', 'insert', 'replace', 'insert_ignore']

    if not cols is None:
        if isinstance(cols, str):
            cols = [cols]
        df = df[cols].copy()
    else:
        cols = df.columns.tolist()

    cur = conn.cursor()

    # 若数据库不存在，则新建
    cur.execute('CREATE DATABASE IF NOT EXISTS {};'.format(db_name))
    if not isnull(db_name):
        cur.execute('USE {};'.format(db_name))

    cols_info, cols_type, cols_info_dict = _get_cols_info_df(df, cols=cols)

    # 表不存在或强制建新表
    if act_type == 'new':
        # 创建table
        cur.execute('DROP TABLE IF EXISTS {};'.format(tb_name))
        cur.execute('CREATE TABLE {a}({b});'.format(a=tb_name, b=cols_info))
    cur.execute('CREATE TABLE IF NOT EXISTS {a}({b});'.format(
                                                    a=tb_name, b=cols_info))
    # 检查字段是否已经存在，不存在新建
    cur.execute('DESC {};'.format(tb_name))
    fields_info = cur.fetchall()
    fields = [x[0] for x in fields_info]
    cols_loss = [x for x in cols if x not in fields]
    for col in cols_loss:
        cur.execute('ALTER TABLE {} ADD {};'.format(
            tb_name, cols_info_dict[col][0]))
    
    # 数据更新
    values = df.values.tolist()
    values = [tuple(x) for x in values]
    if act_type in ['new', 'insert']:
        # 批量插入新数据
        cur.executemany('INSERT INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=cols_type),
                        values)
    elif act_type == 'replace':
        # 批量更新数据
        cur.executemany('REPLACE INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=cols_type),
                        values)
    elif act_type == 'insert_ignore':
        # 批量插入数据，若有重复，保留已存在的不更新
        cur.executemany('INSERT IGNORE INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=cols_type),
                        values)
    
    cur.close()
    conn.commit()
