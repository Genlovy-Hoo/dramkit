# -*- coding: utf-8 -*-

from copy import copy
import numpy as np
import pandas as pd
import pymysql
from pymysql.constants import CLIENT
from dramkit.gentools import (isnull,
                              check_list_arg,
                              df_na2value,
                              change_dict_key,
                              list_eq)
from dramkit.logtools.utils_logger import logger_show


class PyMySQL(object):
    
    def __init__(self,
                 host='localhost', user='root',
                 password=None, database=None,
                 port=3306, logger=None, **kwargs):
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
                             client_flag=CLIENT.MULTI_STATEMENTS,
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
    
    def get_db_info(self, db_names, tb_names=False):
        return get_db_info(self.conn, db_names,
                           tb_names=tb_names)
    
    def get_now_database(self):
        return get_now_database(self.conn)

    def get_data_file_dir(self):
        return get_data_file_dir(self.conn)
    
    def get_split_tabel_info(self, tb_name=None, db_name=None):
        res = get_split_tabel_info(conn=self.conn,
                                   tb_name=tb_name,
                                   db_name=db_name)
        self.reset_db(db_name)
        return res    
    
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
                 db_name=None, nlimit=None):
        df = get_data(conn=self.conn, tb_name=tb_name, cols=cols,
                      where_str=where_str, db_name=db_name,
                      nlimit=nlimit)
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
        
    def get_unique_values(self, tb_name, field, db_name=None):
        res = get_unique_values(self.conn, tb_name, field,
                                db_name=db_name)
        self.reset_db(db_name)
        return res
    
    def df_to_sql(self, df, tb_name, act_type='replace',
                  db_name=None, cols=None, idcols=None,
                  col_types={}, na_val=None,
                  inf_val='inf', _inf_val='-inf',
                  logger=None, **kwargs_cols):
        db_name = self._check_db(db_name)
        logger = self.logger if isnull(logger) else logger
        df_to_sql(df, conn=self.conn, tb_name=tb_name,
                  act_type=act_type, db_name=db_name,
                  cols=cols, idcols=idcols,
                  col_types=col_types, na_val=na_val,
                  inf_val=inf_val, _inf_val=_inf_val,
                  logger=logger, **kwargs_cols)
        self.reset_db(db_name)
        
    def df_to_sql_by_row(self, df, tb_name, act_type='insert',
                         db_name=None, cols=None, idcols=None,
                         col_types={}, na_val=None,
                         inf_val='inf', _inf_val='-inf',
                         logger=None, **kwargs_cols):
        
        db_name = self._check_db(db_name)
        logger = self.logger if isnull(logger) else logger
        df_to_sql_by_row(df, self.conn, tb_name,
                         act_type=act_type, db_name=db_name,
                         cols=cols, idcols=idcols,
                         col_types=col_types, na_val=na_val,
                         inf_val=inf_val, _inf_val=_inf_val,
                         logger=logger, **kwargs_cols)
        self.reset_db(db_name)
        
    def copy(self):
        res = copy(self)
        conn_args = self.__db_conn_args.copy()
        conn_args.update({'client_flag': CLIENT.MULTI_STATEMENTS})
        res.conn = get_conn(**conn_args)
        return res
    
    def has_table(self, tb_name, db_name=None):
        # tbs = self.show_tables(db_name=db_name)
        # return tb_name.lower() in [x.lower() for x in tbs]
        return self.execute_sql('SHOW TABLES LIKE "{}"'.format(tb_name), db_name=db_name).shape[0] > 0

        
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


def execute_sql(conn, sql_str, db_name=None, to_df=True):
    '''执行sql语句并返回结果'''
    conn.ping(reconnect=True)
    cur = conn.cursor()
    if not isnull(db_name):
        cur.execute('USE {};'.format(db_name))
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


def has_table(conn, tb_name, db_name=None):
    # tbs = show_tables(conn, db_name=db_name)
    # return tb_name.lower() in [x.lower() for x in tbs]
    return execute_sql(conn, 'SHOW TABLES LIKE "{}"'.format(tb_name), db_name=db_name).shape[0] > 0


def get_fields(conn, tb_name, db_name=None):
    '''获取表字段名列表'''
    sql_str = 'DESC {};'.format(tb_name)
    fields_info = execute_sql(conn, sql_str,
                              db_name=db_name, to_df=True)
    fields = fields_info['Field'].tolist()
    return fields, fields_info


def get_unique_values(conn, tb_name, field, db_name=None):
    '''获取某个字段的不重复值'''
    sql_str = '''SELECT DISTINCT {} FROM {}
              ;'''.format(field, tb_name)
    res = execute_sql(conn, sql_str, db_name=db_name)
    # res = res[field].tolist()
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


def get_data_file_dir(conn):
    '''查看数据库文件所在位置'''
    return execute_sql(conn, 'SHOW VARIABLES LIKE "datadir";')


def get_split_tabel_info(conn, tb_name=None, db_name=None):
    '''查询表分区信息'''
    where_str = ''
    if not isnull(tb_name):
        where_str = "WHERE table_name='%s'"%tb_name
    if not isnull(db_name):
        where_str = where_str + " AND table_schema='%s'"%db_name
    sql = '''SELECT * 
             FROM information_schema.PARTITIONS 
             {}
          ;'''.format(where_str)
    res = execute_sql(conn, sql)
    return res


def drop_table_split(conn, tb_name, part_names=None,
                     db_name=None):
    '''删除表分区'''
    part_names = _check_list_arg(part_names)
    if isnull(part_names):
        part_names = get_split_tabel_info(conn, tb_name=tb_name, db_name=db_name)
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
    
   
def copy_table(conn, tb_name, db_name=None, with_data=False):
    '''复制表'''
    raise NotImplementedError
    
   
def create_table(conn, tb_name, cols_info, idcols=None,
                 db_name=None, force=False):
    '''
    新建表
    
    Examples
    --------
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> idcols = None
    >>> idcols=['a', 'c']
    >>> create_table(conn, 'test2',
    ...              ('a VARCHAR(255)',
    ...               'b DOUBLE NOT NULL DEFAULT 1',
    ...               'c DATETIME'),
    ...              idcols=idcols,
    ...              db_name='test', force=True)
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
    else:
        colstr = colstr + ')'
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
    
    
def get_db_info(conn, db_names, tb_names=False):
    '''
    | 获取数据库信息
    | tb_names若为False，则结果只统计数据库，不包含单表；
    | 若为None，则包含所有单表；也可以为str或list，指定单个或多个表
    '''
    db_names = check_list_arg(db_names)
    assert isinstance(tb_names, (bool, type(None), str,
                                 list, tuple))
    if tb_names == False:
        sql = '''SELECT 
                     TABLE_SCHEMA "数据库",
                     COUNT(TABLE_NAME) "表数",
                     SUM(TABLE_ROWS) "总记录数",
                     TRUNCATE(SUM(DATA_LENGTH)/1024/1024, 2) "数据容量(MB)",
                     TRUNCATE(SUM(DATA_LENGTH)/1024/1024/1024, 2) "数据容量(GB)",
                     TRUNCATE(SUM(INDEX_LENGTH)/1024/1024, 2) "索引容量(MB)",
                     TRUNCATE(SUM(INDEX_LENGTH)/1024/1024/1024, 2) "索引容量(GB)"
                 FROM 
                     information_schema.TABLES
                 WHERE 
                     TABLE_SCHEMA IN ({})
                 GROUP BY TABLE_SCHEMA
                 ORDER BY SUM(DATA_LENGTH) DESC,
                          SUM(INDEX_LENGTH) DESC
              ;'''.format(', '.join(['"%s"'%x for x in db_names]))
    else:
        tb_names = check_list_arg(tb_names, allow_none=True)
        if isnull(tb_names):
            tb_str = ''
        else:
            tb_str = 'AND TABLE_NAME IN ({})'.format(
                     ', '.join(['"%s"'%x for x in tb_names]))
        sql = '''SELECT 
                     TABLE_SCHEMA "数据库",
                     TABLE_NAME "表名",
                     TABLE_ROWS "记录数",
                     TRUNCATE(DATA_LENGTH/1024/1024, 2) "数据容量(MB)",
                     TRUNCATE(DATA_LENGTH/1024/1024/1024, 2) "数据容量(GB)",
                     TRUNCATE(INDEX_LENGTH/1024/1024, 2) "索引容量(MB)",
                     TRUNCATE(INDEX_LENGTH/1024/1024/1024, 2) "索引容量(GB)"
                 FROM 
                     information_schema.TABLES
                 WHERE 
                     TABLE_SCHEMA IN ({})
                     {}
                 ORDER BY TABLE_SCHEMA ASC,
                          DATA_LENGTH DESC,
                          INDEX_LENGTH DESC
              ;'''.format(', '.join(['"%s"'%x for x in db_names]),
                          tb_str)
    return execute_sql(conn, sql)
    
    
def get_data(conn, tb_name, cols=None, where_str=None,
             db_name=None, nlimit=None):
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
    where_str = '' if isnull(where_str) else 'WHERE %s'%where_str
    limit_str = '' if isnull(nlimit) else 'LIMIT %s'%nlimit
    sql = '''SELECT {}
             FROM {}
             {}
             {}
          ;'''.format(col_str, tb_name, where_str, limit_str)
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


def get_cols_info_df(df, cols=None, col_types={}, 
                     all2str=False, big_text_cols=[]):
    '''
    根据pd.DataFrame中的列cols，识别对应列在MySQL中的字段类别
    
    Parameters
    ----------
    df : pandas.DataFrame
        待识别数据
    cols : list, None
        待识别列名列表，默认所有列
    col_types : dict
        指定列类型，如{'col1': 'VARCHAR(20)', 'col2': 'BIGINT'}，指定的列不做判断，直接返回
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
        占位符信息，格式如'%s, %s, ...'
    
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
    for col in cols:
        if col in col_types:
            char = col + ' ' + col_types[col]
        elif all2str:
            if col in big_text_cols:
                char = col + ' TEXT'
            else:
                char = col + ' VARCHAR(255)'
        elif 'int' in str(types[col]):
            char = col + ' INT'
        elif 'float' in str(types[col]):
            char = col + ' FLOAT' # DOUBLE
        elif 'object' in str(types[col]):
            if col in big_text_cols:
                char = col + ' TEXT'
            else:
                char = col + ' VARCHAR(255)'
        elif 'datetime' in str(types[col]):
            char = col + ' DATETIME'
        else:
            raise ValueError('未识别（暂不支持）的字段类型: %s！'%col)
        char_ = '%s'
        cols_info.append(char)
        placeholder.append(char_)
        cols_info_dict[col] = (char, char_)
    cols_info, placeholder = ', '.join(cols_info), ', '.join(placeholder)
    dtype = {k: v[0].split(' ')[-1] for k, v in cols_info_dict.items()}

    return cols_info, dtype, placeholder, cols_info_dict


def df_to_sql_insert_values(df, cols=None):
    '''
    df转化为mysql插入的VALUES格式
    
    Examples
    --------
    >>> DB = PyMySQL(host='localhost',
    ...              user='root',
    ...              password='xxxxxxxxxxx',
    ...              database='test',
    ...              port=3306,
    ...              logger=None)
    >>> import numpy as np
    >>> df4 = pd.DataFrame({'id1': [1, 2, 3, 4, 5],
    ...                     'id2': [2, 3, 4, 5, 6],
    ...                     'col1': ['a', 'b', 'c', 'd', 'e'],
    ...                     'col2': [2, 4, 6, 8, 10]})
    >>> DB.df_to_sql(df4,'test3', 'new',
    ...              db_name='test', idcols=['id1', 'id2'])
    >>> df = pd.DataFrame({'id1': [6, 7, 8, 9, 10],
    ...                    'id2': [7, 8, 9, 10, 11],
    ...                    'col1': ['f', 'g', 'h', 'i', 'j'],
    ...                    'col2': [3, 6, 9, 12, 15]})
    >>> values = df_to_sql_insert_values(df)
    >>> DB.execute_sql('INSERT INTO test3 VALUES {}'.format(values))
    '''
    cols = df.columns if isnull(cols) else cols
    # values = df[cols].astype(str).values.tolist()
    values = df[cols].values.tolist()
    values = [str(tuple(x)) for x in values]
    values = ', '.join(values)
    return values


def df_to_sql(df, conn, tb_name, act_type='replace',
              db_name=None, cols=None, idcols=None,
              col_types={}, na_val=None, 
              inf_val='inf', _inf_val='-inf',
              logger=None, **kwargs_cols):
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
        |     若为'ignore_tb'，则当表已经存在时不进行任何操作
        |     若为'new'，则新建表（若原表已存在，则会先删除再重建）
        |     若为'clear'，则先清空原表数据（若原表已存在）
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
    >>> db_name = 'test'
    >>> tb_name = 'test1'
    >>> tb_name2 = 'test2'
    >>> df = pd.DataFrame({'code': ['001', '002', '003'],
    ...                    'year': ['2011', '2012', '2013'],
    ...                    'value': [1, 2, 3]})
    >>> df_to_sql(df, conn, tb_name, act_type='new', db_name=db_name)
    >>> df_to_sql(df.rename(columns={'value': 'VALUE'}), conn, tb_name, act_type='replace', db_name=db_name)
    >>> df_to_sql(df, conn, tb_name2, act_type='new',
                  db_name=db_name, idcols=['code', 'year'])
    >>> df2 = pd.DataFrame({'code': ['002', '003', '005'],
    ...                     'value': [2, 4, 5],
    ...                     'year': ['2012', '2014', '2015'],
    ...                     'value2': [3, 5, 6]})
    >>> df_to_sql(df2, conn, tb_name, act_type='insert',
    ...           db_name=db_name, cols='value2')
    >>> df_to_sql(df2, conn, tb_name, act_type='replace',
    ...           db_name=db_name, cols='value2')
    >>> df_to_sql(df, conn, tb_name, act_type='new', db_name=db_name)
    >>> set_primary_key(conn, tb_name, 'code', db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='replace', db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='insert', db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='insert_ignore', db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='insert_ignore', cols='value2', db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='insert_ignore', cols=['code', 'value2'], db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='replace', cols=['code', 'value2'], db_name=db_name)
    >>> df_to_sql(df2, conn, tb_name, act_type='replace', cols=['code', 'year', 'value2'], db_name=db_name)
    >>> modify_cols_type(conn, 'test1', ('code VARCHAR(20)', 'year VARCHAR(10) DEFAULT "XX"', ), db_name)
    >>> df3 = pd.DataFrame({'code': ['006', '007', '008'],
    ...                     'value': [6, 7, 8],
    ...                     'value2': [7, 8, 9]})
    >>> df_to_sql(df3, conn, tb_name, act_type='replace', db_name=db_name)
    ...
    >>> import numpy as np
    >>> conn = get_conn(password='xxxxxxxxxxx')
    >>> df4 = pd.DataFrame({'id1': [1, 2, 3, 4, 5],
    ...                     'id2': [2, 3, 4, 5, 6],
    ...                     'col1': ['a', 'b', 'c', 'd', 'e'],
    ...                     'col2': [2, 4, 6, 8, 10]})
    >>> df5 = pd.DataFrame({'id1': [3, 4, 5, 6, 7],
    ...                     'id2': [4, 5, 6, 7, 8],
    ...                     'col1': ['c', 'ddd', np.nan, 'f', 'g'],
    ...                     'col3': [6, 8, 10, 12, 14]})
    >>> df_to_sql(df4, conn, 'test3', act_type='new',
    ...           db_name='test', idcols=['id1', 'id2'])
    >>> df_to_sql(df5, conn, 'test3',
    ...           act_type='insert',
    ...           db_name='test', idcols=['id1', 'id2'])
    >>> # 测试2
    >>> conn = get_conn(password='xxxxxxxxxxx', database='test')
    >>> tb_name = 'test6'
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
    
    References
    -----------
    - https://blog.csdn.net/qq_43279637/article/details/92797641
    - https://blog.csdn.net/tonydz0523/article/details/82529941
    - https://blog.csdn.net/weixin_44848356/article/details/119113174
    - https://blog.csdn.net/weixin_42272869/article/details/116480732
    '''
    
    assert act_type in ['ignore_tb', 'new', 'insert', 'replace',
                        'insert_ignore', 'insert_newcols',
                        'insert_ignore_newcols']
    if act_type == 'ignore_tb' and has_table(conn, tb_name, db_name=db_name):
        return
    
    idcols = _check_list_arg(idcols)
    if isnull(idcols):
        act_type = 'insert' if act_type != 'new' else 'new'
    
    # 清除原数据或直接新增数据
    if act_type in ['new', 'clear', 'insert', 'insert_ignore']:
        df_to_sql_by_row(df, conn, tb_name, act_type=act_type,
                         db_name=db_name, cols=cols, idcols=idcols,
                         col_types=col_types, na_val=na_val,
                         inf_val=inf_val, _inf_val=_inf_val,
                         logger=logger, **kwargs_cols)
        return
    
    # 待入库字段检查
    cols = _check_list_arg(cols)
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
        
    cur = conn.cursor()

    # 若数据库不存在，则新建
    if not isnull(db_name):
        cur.execute('CREATE DATABASE IF NOT EXISTS {};'.format(db_name))
        cur.execute('USE {};'.format(db_name))

    cols_info, dtype, ph, cols_info_dict = get_cols_info_df(
        df, cols=cols, col_types=change_dict_key(col_types, lambda x: x.upper()),
        **kwargs_cols)

    # 表不存在则新建
    if isnull(idcols):
        cur.execute('CREATE TABLE IF NOT EXISTS {a}({b});'.format(
                     a=tb_name, b=cols_info))
    else:
        idcols = [x.upper() for x in idcols]
        colstr = '('
        colstr = colstr + cols_info.replace(',', ', \n')
        pkstr, ukstr = '', ''
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
    fields = [x[0].upper() for x in fields_info]
    
    # 检查字段是否已经存在，不存在新建
    # TODO: 不存在的字段中有idcols中的字段的情况处理（可以先新建字段再重新设置联合唯一索引）
    cols_loss = [x for x in cols if x not in fields]
    for col in cols_loss:
        cur.execute('ALTER TABLE {} ADD {};'.format(
            tb_name, cols_info_dict[col][0]))
        
    # 唯一值字段
    cur.execute('SHOW INDEX FROM %s WHERE Non_unique = 0;'%tb_name)
    idcols_ = list(set([x[4].upper() for x in cur.fetchall()]))    
    if any([not x in cols for x in idcols_]):
        raise ValueError('待存入数据中必须包含所有唯一值字段！')
        
    # TODO: idcols和idcols_即已有唯一索引字段不符合的情况处理
    assert list_eq(idcols, idcols_, order=False), '`idcols`必须和表唯一值索引相同'
   
    # 先处理已存在字段
    oldcols = [x for x in cols if x in fields]
    oldcols = list(set(idcols + oldcols))
    if len(oldcols) > 0:
        values = df[oldcols].values.tolist()
        colstr = ', '.join(oldcols)
        typestr = ', '.join([cols_info_dict[x][1] for x in oldcols])
        if act_type == 'insert_newcols':
            sql = '''INSERT INTO {} ({}) VALUES ({})
                  ;'''.format(tb_name, colstr, typestr)
        elif act_type == 'replace':
            idstr = ', '.join(['{x} = VALUES({x})'.format(x=x) for x in oldcols])
            sql = '''INSERT INTO {} ({})
                     VALUES ({})
                     ON DUPLICATE KEY UPDATE {}
                  ;'''.format(tb_name, colstr, typestr, idstr)
        else:
            sql = '''INSERT IGNORE INTO {} ({}) VALUES ({})
                  ;'''.format(tb_name, colstr, typestr)
        if act_type == 'insert_newcols':
            try:
                cur.executemany(sql, values)
            except:
                pass
        else:
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


def df_to_sql_by_row(df, conn, tb_name,
                     act_type='insert', db_name=None,
                     cols=None, idcols=None, col_types={},
                     na_val=None, inf_val='inf', _inf_val='-inf',
                     logger=None, **kwargs_cols):
    '''
    把pandas.DataFrame存入MySQL数据库中（不考虑列的新增或缺省）
    
    参数见 :func:`df_to_sql` 
    
    Note
    ----
    判断数据是否存在时是根据主键或唯一索引来的，
    因此当待插入数据字段只是已存在数据字段的一部分时，
    此函数应慎用'replace'（可能导致原有数据变成Null）
    '''
    
    assert act_type in ['ignore_tb', 'new', 'clear', 'insert', 'replace', 'insert_ignore']
    if act_type == 'ignore_tb' and has_table(conn, tb_name, db_name=db_name):
        return
    
    # 待入库字段检查
    cols = _check_list_arg(cols)
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

    cur = conn.cursor()

    # 若数据库不存在，则新建
    if not isnull(db_name):
        cur.execute('CREATE DATABASE IF NOT EXISTS {};'.format(db_name))
        cur.execute('USE {};'.format(db_name))

    cols_info, dtype, ph, cols_info_dict = get_cols_info_df(
        df, cols=cols, col_types=change_dict_key(col_types, lambda x: x.upper()),
        **kwargs_cols)

    # 清空表
    if act_type == 'clear':
        n = cur.execute('SHOW TABLES LIKE "{}";'.format(tb_name))
        if n > 0:
            cur.execute('TRUNCATE TABLE {};'.format(tb_name))
            # cur.execute('DELETE FROM {};'.format(tb_name))
    # 表不存在则新建
    if act_type == 'new':
        cur.execute('DROP TABLE IF EXISTS {};'.format(tb_name))
    idcols = _check_list_arg(idcols)
    if isnull(idcols):
        cur.execute('CREATE TABLE IF NOT EXISTS {a}({b});'.format(
                     a=tb_name, b=cols_info))
    else:
        idcols = [x.upper() for x in idcols]
        colstr = '('
        colstr = colstr + cols_info.replace(',', ', \n')
        pkstr, ukstr = '', ''
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
    fields = [x[0].upper() for x in fields_info]

    cols_loss = [x for x in cols if x not in fields]
    for col in cols_loss:
        cur.execute('ALTER TABLE {} ADD {};'.format(
            tb_name, cols_info_dict[col][0]))
    # 所有字段
    all_fields = list(set(fields + cols_loss))
    
    # 数据更新
    values = df.values.tolist()
    if act_type in ['new', 'clear'] or isnull(idcols):
        cur.executemany('INSERT INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=ph),
                        values)
    elif act_type == 'insert':
        # 批量插入新数据
        try:
            cur.executemany('INSERT INTO {a} ({b}) VALUES ({c});'.format(
                            a=tb_name, b=','.join(cols), c=ph),
                            values)
        except:
            for col in cols_loss:
                cur.execute('ALTER TABLE {} DROP COLUMN {};'.format(tb_name, col))
            raise
    elif act_type == 'replace':
        if len([x for x in all_fields if x not in cols]) > 0:
            logger_show('待存入数据字段不包含表中全部字段，可能导致已有的部分数据丢失！',
                        logger, 'warn')
        # 批量更新数据
        cur.executemany('REPLACE INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=ph),
                        values)
    elif act_type == 'insert_ignore':
        # 批量插入数据，若有重复，保留已存在的不更新
        cur.executemany('INSERT IGNORE INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=ph),
                        values)
    
    cur.close()
    conn.commit()
    
    
def gen_simple_proc_sql_str(sql_list, proc_name=None,
                            with_call=True, delimiter=False):
    '''
    生成存储过程mysql语句
    
    Note
    ----
    - DELIMITER语句在pymysql中执行会报错，不知道为啥？故若要在pymysql中执行生成的sql语句，delimiter应设置为False
    - ROLLBACK只对DML语句(INSERT、DELETE、UPDATE，SELECT除外)起作用
    - (DDL语句(Data Definition Language)：CREATE、ALTER、DROP)
    - (DML(Data Manipulation Language)：INSERT、DELETE、UPDATE、SELECT)
    - (DCL语句(Data Control Language)：GRANT、ROLLBACK、COMMIT)
    
    Examples
    --------
    >>> DB = PyMySQL(host='localhost',
    ...              user='root',
    ...              password='xxxxxxxxxxx',
    ...              database=None,
    ...              port=3306,
    ...              logger=None)
    >>> import numpy as np
    >>> df4 = pd.DataFrame({'id1': [1, 2, 3, 4, 5],
    ...                     'id2': [2, 3, 4, 5, 6],
    ...                     'col1': ['a', 'b', 'c', 'd', 'e'],
    ...                     'col2': [2, 4, 6, 8, 10]})
    >>> DB.df_to_sql(df4,'test3', 'new',
    ...              db_name='test', idcols=['id1', 'id2'])
    >>> delimiter = False
    >>> sql_list = ['ALTER TABLE test3 ADD col3 VARCHAR(10);',
    ...             'INSERT INTO test3 VALUES (8, 9, 10, 11, 12);']
    >>> sql = gen_simple_proc_sql_str(sql_list, delimiter=delimiter)
    >>> # 执行产生的sql，test3表会新增一列和一行
    >>> sql_list = ['ALTER TABLE test3 ADD col4 VARCHAR(10);',
    ...             'INSERT INTO test3 VALUES (8, 9, 10, 11, 12, 13);']
    >>> sql = gen_simple_proc_sql_str(sql_list, delimiter=delimiter)
    >>> # 执行产生的sql，test3表增加一列（ROLLBACK对增加字段操作无效）
    >>> sql_list = ['INSERT INTO test3 VALUES (9, 10, 11, 12, 14, 14);',
    ...             'INSERT INTO test3 VALUES (8, 9, 10, 11, 12, 13);']
    >>> sql = gen_simple_proc_sql_str(sql_list, delimiter=delimiter)
    >>> # 执行产生的sql，test3表不会变
    >>> sql_list = ['INSERT INTO test3 VALUES (9, 10, 11, 12, 14, 14);',
    ...             'INSERT INTO test3 VALUES (10, 11, 12, 13, 14, 15);']
    >>> sql = gen_simple_proc_sql_str(sql_list, delimiter=delimiter)
    >>> # 执行产生的sql，test3表会增加两行
    
    References
    ----------
    - https://www.cnblogs.com/rendd/p/16596352.html
    - https://blog.csdn.net/qq_34745941/article/details/115733016
    - https://www.cnblogs.com/yunlong-study/p/14441447.html
    - https://blog.csdn.net/weixin_34551601/article/details/113910378
    '''
    sqlstrs = _check_list_arg(sql_list, allow_none=False)
    if isnull(proc_name):
        proc_name = 'xxnew_procxx'
    if not with_call:
        call_str = '''        
        /*执行*/
        -- CALL {x}();
    
        /*删除存储过程*/
        -- DROP PROCEDURE IF EXISTS {x};
        '''.format(x=proc_name)
    else:
        call_str = '''        
        /*执行*/
        CALL {x}();
    
        /*删除存储过程*/
        DROP PROCEDURE IF EXISTS {x};
        '''.format(x=proc_name)
    assert delimiter in [None, False] or isinstance(delimiter, str)
    if isnull(delimiter):
        delimiter = '$$'
    if not delimiter:
        delimiter1 = ''
        delimiter2 = 'END;'
        delimiter3 = ''
    else:
        delimiter1 = '''
        /*声明结束符*/
        DELIMITER {}        
        '''.format(delimiter)
        delimiter2 = 'END {}'.format(delimiter)
        delimiter3 = '''            
        /*还原结束符为`;`*/
        DELIMITER ;        
        '''
    sqlstr = '''        
        {}
        /*若存在，则删除存储过程*/
        DROP PROCEDURE IF EXISTS {};
        
        /*创建存储过程*/
        CREATE PROCEDURE {}()
            BEGIN
                /*声明一个变量，标识是否有sql异常*/
                DECLARE hasSqlError int DEFAULT FALSE;
                /*在执行过程中出任何异常设置hasSqlError为TRUE*/
                DECLARE CONTINUE HANDLER FOR SQLEXCEPTION SET hasSqlError=TRUE;
            
                /*开启事务*/
                START TRANSACTION;
                
                /*事务代码*/
                {}
            
                /*根据hasSqlError判断是否有异常，做回滚和提交操作*/
                IF hasSqlError THEN
                    ROLLBACK;
                ELSE
                    COMMIT;
                END IF;
            {}
        {}
        {}        
        /*复制以上返回代码到MySQL中执行以创建&执行存储过程*/
        '''.format(delimiter1, proc_name, proc_name,
                   '\n                '.join(sqlstrs),
                   delimiter2, delimiter3, call_str)
    return sqlstr
