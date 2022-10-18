# -*- coding: utf-8 -*-

import pymysql
import pandas as pd
from dramkit.gentools import isnull
from dramkit.logtools.utils_logger import logger_show


class PyMySQL(object):
    
    def __init__(self,
                 host='localhost', user='root',
                 password=None, database=None,
                 port=3306, logger=None, **kwargs):
        self.conn = get_conn(host=host,
                             user=user,
                             password=password,
                             database=database,
                             port=port,
                             **kwargs)
        self.db_name_ori = database
        self.db_name = database
        self.logger = logger
    
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
    
    def show_tables(self, db_name=None):
        tbs = show_tables(conn=self.conn,
                          db_name=db_name)
        self.reset_db(db_name)
        return tbs
    
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
    
    def get_fields(self, tb_name, db_name=None):
        fields, fields_info = get_fields(conn=self.conn,
                                         tb_name=tb_name,
                                         db_name=db_name)
        self.reset_db(db_name)
        return fields, fields_info
    
    def get_id_fields(self, tb_name, db_name=None):
        idcols = get_id_fields(conn=self.conn,
                               tb_name=tb_name,
                               db_name=db_name)
        self.reset_db(db_name)
        return idcols
    
    def get_now_database(self):
        return get_now_database(self.conn)
    
    def get_split_tabel_info(self, tb_name=None):
        return get_split_tabel_info(conn=self.conn, tb_name=tb_name)
    
    def drop_table_split(self, tb_name, part_names=None,
                         db_name=None):
        drop_table_split(conn=self.conn, tb_name=tb_name,
                         part_names=part_names, db_name=db_name)
        self.reset_db(db_name)
        
    def cancel_split_table(self, tb_name, db_name=None):
        cancel_split_table(conn=self.conn, tb_name=tb_name,
                           db_name=db_name)
        self.reset_db(db_name)
    
    def drop_index(self, tb_name, index_name, db_name=None):
        drop_index(conn=self.conn, tb_name=tb_name,
                   index_name=index_name, db_name=db_name)
        self.reset_db(db_name)
        
    def drop_primary_key(self, tb_name, db_name=None):
        drop_primary_key(conn=self.conn, tb_name=tb_name,
                         db_name=db_name)
        self.reset_db(db_name)
        
    def drop_table(self, tb_name, db_name=None):
        drop_table(conn=self.conn, tb_name=tb_name, db_name=db_name)
        self.reset_db(db_name)
        
    def create_database(self, db_name):
        create_database(self.conn, db_name)
        
    def drop_database(self, db_name):
        drop_database(self.conn, db_name)
        
    def create_table(self, tb_name, cols_info, idcols=None,
                     db_name=None, force=False):
        create_table(conn=self.conn, tb_name=tb_name,
                     cols_info=cols_info, idcols=idcols,
                     db_name=db_name, force=force)
        self.reset_db(db_name)
    
    def add_cols(self, tb_name, cols_info, db_name=None):
        add_cols(conn=self.conn, tb_name=tb_name,
                 cols_info=cols_info, db_name=db_name)
        self.reset_db(db_name)
        
    def modify_cols_type(self, tb_name, cols_info, db_name=None):
        modify_cols_type(conn=self.conn, tb_name=tb_name,
                         cols_info=cols_info, db_name=db_name)
        self.reset_db(db_name)
        
    def change_cols_info(self, tb_name, cols_info, db_name=None):
        change_cols_info(conn=self.conn, tb_name=tb_name,
                        cols_info=cols_info, db_name=db_name)
        self.reset_db(db_name)
        
    def drop_cols(self, tb_name, cols, db_name=None):
        drop_cols(conn=self.conn, tb_name=tb_name,
                  cols=cols, db_name=db_name)
        self.reset_db(db_name)
        
    def clear_data(self, tb_name, db_name=None):
        clear_data(conn=self.conn, tb_name=tb_name, db_name=db_name)
        self.reset_db(db_name)
        
    def drop_data_by_where_str(self, tb_name, where_str,
                               db_name=None):
        drop_data_by_where_str(conn=self.conn, tb_name=tb_name,
                               where_str=where_str, db_name=db_name)
        self.reset_db(db_name)
        
    def get_data(self, tb_name, cols=None, where_str=None,
                 db_name=None):
        df = get_data(conn=self.conn, tb_name=tb_name, cols=cols,
                      where_str=where_str, db_name=db_name)
        self.reset_db(db_name)
        return df
    
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
                    cols=None, db_name=None, na_val=None,
                    idcols=None, logger=None,
                    **kwargs_cols):
        db_name = self._check_db(db_name)
        logger = self.logger if isnull(logger) else logger
        df_to_mysql(df, conn=self.conn, tb_name=tb_name,
                    act_type=act_type, cols=cols,
                    db_name=db_name, na_val=na_val,
                    idcols=idcols, logger=logger,
                    **kwargs_cols)
        self.reset_db(db_name)
        
        
def _check_list_arg(arg, allow_none=True):
    if allow_none:
        assert isinstance(arg, (type(None), str, list, tuple))
    else:
        assert isinstance(arg, (str, list, tuple))
    if isinstance(arg, str):
        arg = [arg]
    return arg


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


def get_fields(conn, tb_name, db_name=None):
    '''获取表字段名列表'''
    sql_str = 'DESC {};'.format(tb_name)
    fields_info = execute_sql(conn, sql_str,
                              db_name=db_name, to_df=True)
    fields = fields_info['Field'].tolist()
    return fields, fields_info


def execute_sql(conn, sql_str, db_name=None, to_df=True):
    '''执行sql语句并返回结果'''
    cur = conn.cursor()
    if not isnull(db_name):
        cur.execute('USE {};'.format(db_name))
    cur.execute('SELECT DATABASE();')
    cur.execute(sql_str)
    res = cur.fetchall()
    if isnull(cur.description):
        cur.close()
        conn.commit()
        return None
    if to_df:
        cols = [x[0] for x in cur.description]
        res = pd.DataFrame(res, columns=cols)
    cur.close()
    conn.commit()
    return res


def show_tables(conn, db_name=None):
    '''查看已有表名'''
    res = execute_sql(conn, 'SHOW TABLES;',
                      db_name=db_name, to_df=False)
    return [x[0] for x in res]


def get_now_database(conn):
    '''查询当前选择的数据库'''
    db = execute_sql(conn, 'SELECT DATABASE();', to_df=False)
    return db[0][0]


def get_split_tabel_info(conn, tb_name=None):
    '''查询表分区信息'''
    tb_str = ''
    if not isnull(tb_name):
        tb_str = "WHERE table_name='%s'"%tb_name
    sql = '''SELECT * 
             FROM information_schema.PARTITIONS 
             {}
          ;'''.format(tb_str)
    res = execute_sql(conn, sql)
    return res


def drop_table_split(conn, tb_name, part_names=None,
                     db_name=None):
    '''删除表分区'''
    part_names = _check_list_arg(part_names)
    if isnull(part_names):
        part_names = get_split_tabel_info(conn, tb_name=tb_name)
        part_names = part_names['PARTITION_NAME'].tolist()
    for pname in part_names:
        sql = 'ALTER TABLE %s DROP PARTITION %s;'%(tb_name, pname)
        execute_sql(conn, sql, db_name=db_name)
        
        
def cancel_split_table(conn, tb_name, db_name=None):
    '''取消表分区'''
    sql = 'ALTER TABLE %s REMOVE PARTITIONING'%tb_name
    execute_sql(conn, sql, db_name=db_name)


def get_primary_keys(conn, tb_name, db_name=None):
    '''获取主键列名'''
    db_str = ''
    if not isnull(db_name):
        db_str = "AND CONSTRAINT_SCHEMA='{}'".format(db_name)
    sql = '''SELECT column_name
             FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
             WHERE constraint_name='PRIMARY'
             AND table_name='{}' 
             {}              
          ;'''.format(tb_name, db_str)
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


def get_id_fields(conn, tb_name, db_name=None):
    '''获取表中的唯一值字段列表'''
    sql = '''SHOW INDEX FROM {}
              WHERE Non_unique = 0;
          '''.format(tb_name)
    idcols = execute_sql(conn, sql, db_name=db_name, to_df=False)
    idcols = list(set([x[4] for x in idcols]))
    return idcols


def set_primary_key(conn, tb_name, cols_key, db_name=None):
    '''主键设置'''
    cols_key = _check_list_arg(cols_key, allow_none=False)
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
    cols_uni = _check_list_arg(cols_uni, allow_none=False)
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
    
    
def drop_index(conn, tb_name, index_name, db_name=None):
    '''删除索引'''
    sql_str = 'DROP INDEX {} ON {};'.format(index_name, tb_name)
    execute_sql(conn, sql_str, db_name=db_name, to_df=False)
    
    
def drop_primary_key(conn, tb_name, db_name=None):
    '''删除主键'''
    sql = 'ALTER TABLE {} DROP PRIMARY KEY;'.format(tb_name)
    execute_sql(conn, sql, db_name=db_name, to_df=False)
    
    
def create_database(conn, db_name):
    '''新建数据库'''
    execute_sql(conn, 'CREATE DATABASE IF NOT EXISTS %s;'%db_name)
    
   
def create_table(conn, tb_name, cols_info, idcols=None,
                 db_name=None, force=False):
    '''
    新建表
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> create_table(conn, 'test2',
    ...              ('a VARCHAR(255)',
    ...               'b DOUBLE NOT NULL DEFAULT 1',
    ...               'c DATETIME'),
    ...              idcols=['a', 'c'],
    ...              db_name='test')
    '''
    if force:
        execute_sql(conn, 'DROP TABLE IF EXISTS %s;'%tb_name,
                    db_name=db_name, to_df=False)
    idcols = _check_list_arg(idcols)
    colstr = '('
    colstr = colstr + ', \n'.join(cols_info)
    pkstr, ukstr = '', ''
    if not isnull(idcols):
        pkstr = '\nPRIMARY KEY {} ({}),'.format(tb_name, ', '.join(idcols))
        ukstr = '\nUNIQUE INDEX {} ({})'.format(tb_name, ', '.join(idcols))
    colstr = colstr + ',' + pkstr + ukstr + ')'
    sql = '''CREATE TABLE IF NOT EXISTS {}
             {}
          ;'''.format(tb_name, colstr)
    execute_sql(conn, sql, db_name=db_name, to_df=False)
    
    
def drop_database(conn, db_name):
    '''删除数据库'''
    execute_sql(conn, 'DROP DATABASE IF EXISTS %s;'%db_name)
    
    
def drop_table(conn, tb_name, db_name=None):
    '''删除表'''
    sql_str = 'DROP TABLE IF EXISTS {};'.format(tb_name)
    execute_sql(conn, sql_str, db_name=db_name, to_df=False)
    
    
def drop_cols(conn, tb_name, cols, db_name=None):
    '''删除字段'''
    cols = _check_list_arg(cols, allow_none=False)
    colstr = ', \n'.join(['DROP COLUMN %s'%c for c in cols])
    sql_str = '''ALTER TABLE {}
                {}
              ;'''.format(tb_name, colstr)
    execute_sql(conn, sql_str, db_name=db_name, to_df=False)
    
    
def clear_data(conn, tb_name, db_name=None):
    '''清空数据'''
    sql = 'TRUNCATE TABLE {};'.format(tb_name)
    execute_sql(conn, sql, db_name=db_name, to_df=False)
    
    
def get_data(conn, tb_name, cols=None, where_str=None,
             db_name=None):
    '''
    获取数据
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> get_data(conn, 'test1', cols=None,
    ...          where_str='year = "2012"',
    ...          db_name='test')
    >>> get_data(conn, 'test1', cols='value',
    ...          where_str='value2 IS NULL',
    ...          db_name='test')
    '''
    cols = _check_list_arg(cols)
    col_str = '*' if isnull(cols) else ', '.join(cols)
    where_str = ' ' if isnull(where_str) else 'WHERE %s'%where_str
    sql = 'SELECT {} FROM {}{};'.format(col_str, tb_name, where_str)
    return execute_sql(conn, sql, db_name=db_name)


def get_data_tables(conn, tb_cols_dict, join_cols,
                    db_name=None):
    '''联表查数据'''
    raise NotImplementedError
    
    
def drop_data_by_where_str(conn, tb_name, where_str,
                           db_name=None):
    '''
    删除数据，where_str为where条件语句
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> drop_data_by_where_str(conn, 'test1',
    ...                        'year = "2012"',
    ...                        db_name='test')
    >>> drop_data_by_where_str(conn, 'test1',
    ...                        'value2 IS NULL',
    ...                        db_name='test')
    '''
    sql = 'DELETE FROM {} WHERE {};'.format(
                                        tb_name, where_str)
    execute_sql(conn, sql, db_name=db_name, to_df=False)
    
    
def add_cols(conn, tb_name, cols_info, db_name=None):
    '''
    新增字段
    
    TODO
    ----
    在指定位置处插入新的列？
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> add_cols(conn, 'test1',
    ...          ('a VARCHAR(255)',
    ...           'b DOUBLE NOT NULL DEFAULT 1',
    ...           'c DATETIME'),
    ...          db_name='test')
    '''
    colstr = ', \n'.join(['ADD {}'.format(x) for x in cols_info])
    sql = '''ALTER TABLE {}
             {}
          ;'''.format(tb_name, colstr)
    execute_sql(conn, sql, db_name=db_name, to_df=False)
    
    
def modify_cols_type(conn, tb_name, cols_info, db_name=None):
    '''
    更改字段属性
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> modify_cols_type(conn, 'test1',
    ...                  ('code VARCHAR(21)',
    ...                   'year VARCHAR(20) DEFAULT "XX"'),
    ...                  db_name='test')
    '''
    colstr = ', \n'.join(['MODIFY {}'.format(x) for x in cols_info])
    sql = '''ALTER TABLE {}
             {}
          ;'''.format(tb_name, colstr)
    execute_sql(conn, sql, db_name=db_name, to_df=False)
    
    
def change_cols_info(conn, tb_name, cols_info, db_name=None):
    '''修改字段信息，如重命名，修改字段类型等'''
    '''
    更改字段属性
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> change_cols_info(conn, 'test1',
                         ('code1 code VARCHAR(22)',
                          'year0 year VARCHAR(22) DEFAULT "XX"'),
                         db_name='test')
    '''
    colstr = ', \n'.join(['CHANGE {}'.format(x) for x in cols_info])
    sql = '''ALTER TABLE {}
             {}
          ;'''.format(tb_name, colstr)
    execute_sql(conn, sql, db_name=db_name, to_df=False)


def get_cols_info_df(df, cols=None, big_text=False):
    '''
    根据pd.DataFrame中的列cols，识别对应列在MySQL中的字段类别
    
    TODO
    ----
    - 是否为长文本分字段单独设置
    - 单独设置每列的类型或指定某些类型需要特殊处理的列
    
    Parameters
    ----------
    df : pandas.DataFrame
        待识别数据
    cols : list, None
        待识别列名列表，默认所有列
    big_text : bool
        文本是否为长文本，若为True，则文本保存为TEXT类型
        
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
            char = col + ' DOUBLE'
            char_ = '%s'
        elif 'object' in str(types[col]):
            if big_text:
                char = col + ' TEXT'
            else:
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
                cols=None, db_name=None, na_val=None,
                idcols=None, logger=None, **kwargs_cols):
    '''
    把pandas.DataFrame存入MySQL数据库中
    
    Parameters
    ----------
    df : pandas.DataFrame
        待存数据
    conn : pymysql.connect
        pymysql.connect数据库连接对象
    tb_name : str
        存入的表名
    act_type : str
        | 存入方式：
        |     若为'new'，则新建表（若原表已存在，则会清空已有数据）
        |     若为'insert'，则直接插入
        |     若为'replace'，则将已存在的数据更新，不存在的新插入
        |     若为'insert_ignore'，则已有的不更新，不存在的新插入
    cols : None, list
        需要存入的数据列
    db_name : str
        存入的数据库名
    na_val : None, bool, str
        设置df中na值的填充值
    idcols : None, str, list
        设置唯一值标识字段，只有在表不存在需要新建的时候才起作用
        
    Examples
    --------
    >>> db_name = 'test'
    >>> tb_name = 'test1'
    >>> tb_name2 = 'test2'
    >>> df = pd.DataFrame({'code': ['001', '002', '003'],
    ...                    'year': ['2011', '2012', '2013'],
    ...                    'value': [1, 2, 3]})
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> df_to_mysql(df, conn, tb_name, 'new', db_name=db_name)
    >>> df_to_mysql(df, conn, tb_name2, act_type='new',
                    db_name=db_name, idcols=['code', 'year'])
    >>> df2 = pd.DataFrame({'code': ['002', '003', '005'],
    ...                     'value': [2, 4, 5],
    ...                     'year': ['2012', '2014', '2015'],
    ...                     'value2': [3, 5, 6]})
    >>> df_to_mysql(df2, conn, tb_name, 'insert', db_name=db_name,
    ...             cols='value2')
    >>> df_to_mysql(df2, conn, tb_name, 'replace', db_name=db_name,
    ...             cols='value2')
    >>> df_to_mysql(df, conn, tb_name, 'new', db_name=db_name)
    >>> set_primary_key(conn, tb_name, 'code', db_name=db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'replace', db_name=db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'insert', db_name=db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'insert_ignore', db_name=db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'insert_ignore', 'value2', db_name=db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'insert_ignore', ['code', 'value2'], db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'replace', ['code', 'value2'], db_name)
    >>> df_to_mysql(df2, conn, tb_name, 'replace', ['code', 'year', 'value2'], db_name)
    >>> modify_cols_type(conn, 'test1', ('code VARCHAR(20)', 'year VARCHAR(10) DEFAULT "XX"', ), db_name)
    >>> df3 = pd.DataFrame({'code': ['006', '007', '008'],
    ...                     'value': [6, 7, 8],
    ...                     'value2': [7, 8, 9]})
    >>> df_to_mysql(df3, conn, tb_name, 'replace', db_name=db_name)
    
    References
    -----------    
    - https://blog.csdn.net/tonydz0523/article/details/82529941
    - https://blog.csdn.net/weixin_44848356/article/details/119113174
    - https://blog.csdn.net/weixin_42272869/article/details/116480732
    '''
    
    assert act_type in ['new', 'insert', 'replace', 'insert_ignore']
    
    # 清除原数据或直接新增数据
    if act_type in ['new', 'insert']:
        df_to_mysql_by_row(df, conn, tb_name, act_type=act_type,
                           cols=cols, db_name=db_name, na_val=na_val,
                           idcols=idcols, logger=logger, **kwargs_cols)
        return
    
    # 待入库字段检查
    cols = _check_list_arg(cols)
    if not cols is None:
        df = df[cols].copy()
    else:
        cols = df.columns.tolist()
        
    if na_val != False:
        df = df.where(df.notna(), na_val)
        
    cur = conn.cursor()

    # 若数据库不存在，则新建
    if not isnull(db_name):
        cur.execute('CREATE DATABASE IF NOT EXISTS {};'.format(db_name))
        cur.execute('USE {};'.format(db_name))

    cols_info, cols_type, cols_info_dict = get_cols_info_df(df, cols=cols, **kwargs_cols)

    # 表不存在则新建
    idcols = _check_list_arg(idcols)
    if isnull(idcols):
        cur.execute('CREATE TABLE IF NOT EXISTS {a}({b});'.format(
                     a=tb_name, b=cols_info))
    else:
        colstr = '('
        colstr = colstr + cols_info.replace(',', ', \n')
        pkstr, ukstr = '', ''
        if not isnull(idcols):
            pkstr = '\nPRIMARY KEY {} ({}),'.format(tb_name, ', '.join(idcols))
            ukstr = '\nUNIQUE INDEX {} ({})'.format(tb_name, ', '.join(idcols))
        colstr = colstr + ',' + pkstr + ukstr + ')'
        sql = '''CREATE TABLE IF NOT EXISTS {}
                 {}
              ;'''.format(tb_name, colstr)
        cur.execute(sql)
        
    # 表字段列表
    cur.execute('DESC {};'.format(tb_name))
    fields_info = cur.fetchall()
    fields = [x[0] for x in fields_info]
    
    # 检查字段是否已经存在，不存在新建
    cols_loss = [x for x in cols if x not in fields]
    for col in cols_loss:
        cur.execute('ALTER TABLE {} ADD {};'.format(
            tb_name, cols_info_dict[col][0]))
        
    # 唯一值字段
    cur.execute('SHOW INDEX FROM %s WHERE Non_unique = 0;'%tb_name)
    idcols = list(set([x[4] for x in cur.fetchall()]))
    
    if any([not x in cols for x in idcols]):
        raise ValueError('待存入数据中必须包含所有唯一值字段！')
   
    # 先处理已存在字段
    oldcols = [x for x in cols if x in fields]
    oldcols = list(set(idcols + oldcols))
    if len(oldcols) > 0:
        values = df[oldcols].values.tolist()
        colstr = ', '.join(oldcols)
        typestr = ', '.join([cols_info_dict[x][1] for x in oldcols])
        if act_type == 'replace':
            idstr = ', '.join(['{x} = VALUES({x})'.format(x=x) for x in oldcols])
            sql = '''INSERT INTO {} ({})
                     VALUES ({})
                     ON DUPLICATE KEY UPDATE {}
                  ;'''.format(tb_name, colstr, typestr, idstr)
        else:
            sql = '''INSERT IGNORE INTO {} ({}) VALUES ({})
                  ;'''.format(tb_name, colstr, typestr)
        cur.executemany(sql, values)
        
    # 再处理新增字段
    if len(cols_loss) > 0:
        newcols = list(set(idcols + cols_loss))
        values = df[newcols].values.tolist()
        colstr = ', '.join(newcols)
        typestr = ', '.join([cols_info_dict[x][1] for x in newcols])
        idstr = ', '.join(['{x} = VALUES({x})'.format(x=x) for x in newcols])
        sql = '''INSERT INTO {} ({})
                 VALUES ({})
                 ON DUPLICATE KEY UPDATE {}
              ;'''.format(tb_name, colstr, typestr, idstr)
        cur.executemany(sql, values)
    
    cur.close()
    conn.commit()


def df_to_mysql_by_row(df, conn, tb_name, act_type='insert',
                       cols=None, db_name=None, na_val=None,
                       idcols=None, logger=None, **kwargs_cols):
    '''
    把pandas.DataFrame存入MySQL数据库中（不考虑列的新增或缺省）
    
    参数见 :func:`df_to_mysql` 
    
    Note
    ----
    判断数据是否存在时是根据主键或唯一索引来的，
    因此当待插入数据字段只是已存在数据字段的一部分时，
    此函数应慎用'replace'（可能导致原有数据变成Null）
    '''
    
    assert act_type in ['new', 'insert', 'replace', 'insert_ignore']
    # 待入库字段检查
    cols = _check_list_arg(cols)
    if not cols is None:
        df = df[cols].copy()
    else:
        cols = df.columns.tolist()
        
    if na_val != False:
        df = df.where(df.notna(), na_val)

    cur = conn.cursor()

    # 若数据库不存在，则新建
    if not isnull(db_name):
        cur.execute('CREATE DATABASE IF NOT EXISTS {};'.format(db_name))
        cur.execute('USE {};'.format(db_name))

    cols_info, cols_type, cols_info_dict = get_cols_info_df(df, cols=cols, **kwargs_cols)

    # 表不存在则新建
    idcols = _check_list_arg(idcols)
    if isnull(idcols):
        cur.execute('CREATE TABLE IF NOT EXISTS {a}({b});'.format(
                     a=tb_name, b=cols_info))
    else:
        colstr = '('
        colstr = colstr + cols_info.replace(',', ', \n')
        pkstr, ukstr = '', ''
        if not isnull(idcols):
            pkstr = '\nPRIMARY KEY {} ({}),'.format(tb_name, ', '.join(idcols))
            ukstr = '\nUNIQUE INDEX {} ({})'.format(tb_name, ', '.join(idcols))
        colstr = colstr + ',' + pkstr + ukstr + ')'
        sql = '''CREATE TABLE IF NOT EXISTS {}
                 {}
              ;'''.format(tb_name, colstr)
        cur.execute(sql)
    
    # 表字段列表
    cur.execute('DESC {};'.format(tb_name))
    fields_info = cur.fetchall()
    fields = [x[0] for x in fields_info]
    
    if act_type == 'new':
        # 清空已存在数据
        cur.execute('TRUNCATE TABLE {};'.format(tb_name))
        # 删除多余列
        ecols = [x for x in fields if x not in cols]
        if len(ecols) > 0:
            colstr = ', \n'.join(['DROP COLUMN %s'%c for c in ecols])
            sql = '''ALTER TABLE {}
                     {}
                  ;'''.format(tb_name, colstr)
            cur.execute(sql)

    # 检查字段是否已经存在，不存在新建
    cols_loss = [x for x in cols if x not in fields]
    for col in cols_loss:
        cur.execute('ALTER TABLE {} ADD {};'.format(
            tb_name, cols_info_dict[col][0]))
    # 所有字段
    all_fields = list(set(fields + cols_loss))
    
    # 数据更新
    values = df.values.tolist()
    if act_type in ['new', 'insert']:
        # 批量插入新数据
        cur.executemany('INSERT INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=cols_type),
                        values)
    elif act_type == 'replace':
        if len([x for x in all_fields if x not in cols]) > 0:
            logger_show('待存入数据字段不包含表中全部字段，可能导致已有部分数据丢失！',
                        logger, 'warn')
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
