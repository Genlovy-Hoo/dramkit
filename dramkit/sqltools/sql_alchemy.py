# -*- coding: utf-8 -*-

import warnings
from copy import copy
import pandas as pd
from sqlalchemy import text, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.types import NVARCHAR, DATE, FLOAT, INT, CLOB
from dramkit.gentools import (isnull,
                              change_dict_key,
                              get_update_kwargs)


class SQLAlchemy(object):
    
    def __init__(self,
                 dialect='mysql',
                 driver='pymysql',
                 username='root',
                 password=None,
                 host='localhost',
                 port=3306,
                 database=None,
                 orcl_pdb=False,
                 **kwargs):
        self.__db_conn_args = {'dialect': dialect,
                               'driver': driver,
                               'host': host,
                               'username': username,
                               'password': password,
                               'database': database,
                               'port': port,
                               'orcl_pdb': orcl_pdb
                               }
        self.__db_conn_args.update(kwargs)
        self.dbtype = dialect.lower()
        self.conn = get_conn(dialect=dialect,
                             driver=driver,
                             username=username,
                             password=password,
                             host=host,
                             port=port,
                             database=database,
                             orcl_pdb=orcl_pdb,
                             **kwargs)
        self.db_name = database
        self.execute_sql = self._execute_sql if not orcl_pdb else self._execute_sql_orm
        
    def df_to_sql(self, *args, **kwargs):
        if self.dbtype == 'oracle':
            return self.df_to_oracle(*args, **kwargs)
        elif self.dbtype == 'mysql':
            # from dramkit.sqltools.py_mysql import df_to_sql
            raise NotImplementedError
        
    def _execute_sql(self, sql_str, to_df=True):
        with self.conn.connect() as conn:
            try:
                res = conn.execute(text(sql_str))
            except:
                # warnings.filterwarnings('ignore')
                res = conn.execute(sql_str)
                # warnings.filterwarnings('default')
            try:
                conn.commit()
            except:
                pass
            if isnull(res.cursor):
                return None
            if to_df:
                cols = [x[0] for x in res.cursor.description]
                return pd.DataFrame(res, columns=cols)
            return res
        
    def _execute_sql_orm(self, sql_str, to_df=True):
        with Session(self.conn) as sess:
            try:
                res = sess.execute(text(sql_str))
            except:
                # res = sess.execute(sql_str)
                res = sess.execute(text(sql_str))
            try:
                sess.commit()
            except:
                pass
            if isnull(res.cursor):
                return None
            if to_df:
                cols = [x[0] for x in res.cursor.description]
                return pd.DataFrame(res, columns=cols)
            else:
                return res.cursor.fetchall()
        
    def copy(self):
        res = copy(self)
        res.conn = get_conn(**self.__db_conn_args)
        return res
        
    def get_tables(self):
        if self.dbtype == 'oracle':
            df = self.execute_sql('SELECT * FROM user_tables')
            tbs = df['TABLE_NAME'].tolist()
            return tbs, df
        elif self.dbtype == 'mysql':
            res = self.execute_sql('SHOW TABLES;', to_df=False)
            return [x[0] for x in res], None
        
    def has_table(self, tb_name):
        if self.dbtype == 'oracle':
            sql = "SELECT COUNT(*) n FROM user_tables WHERE table_name=UPPER('{}')".format(tb_name)
            n = self.execute_sql(sql, to_df=False)[0][0]
            if n > 0:
                return True
            return False
        elif self.dbtype == 'mysql':
            tbs = self.get_tables()
            return tb_name.lower() in [x.lower() for x in tbs]
        
    def drop_table(self, tb_name, purge=True):
        '''删除表'''
        if self.dbtype == 'oracle':
            if self.has_table(tb_name):
                if purge:
                    self.execute_sql('DROP TABLE %s PURGE'%tb_name)
                else:
                    self.execute_sql('DROP TABLE %s'%tb_name)
        elif self.dbtype == 'mysql':
            sql = 'DROP TABLE IF EXISTS {};'.format(tb_name)
            self.execute_sql(sql, to_df=False)
        
    def get_fields(self, tb_name):
        '''获取表字段名列表'''
        if self.dbtype == 'oracle':
            sql_str = '''SELECT COLUMN_NAME, DATA_TYPE,
                                DATA_LENGTH, NULLABLE
                         FROM user_tab_columns
                         WHERE table_name = UPPER('{}')
                      '''.format(tb_name)
            fields_info = self.execute_sql(sql_str, to_df=True)
            fields = fields_info['COLUMN_NAME'].tolist()
            info_dict = fields_info.set_index('COLUMN_NAME')['DATA_TYPE'].to_dict()
            return fields, fields_info, info_dict
        elif self.dbtype == 'mysql':
            sql_str = 'DESC {};'.format(tb_name)
            fields_info = self.execute_sql(sql_str, to_df=True)
            fields = fields_info['Field'].tolist()
            info_dict = fields_info.set_index('Field')['Type'].to_dict()
            return fields, fields_info, info_dict
    
    def _get_tmp_tb_name(self, tmp_tb_name):
        assert isinstance(tmp_tb_name, str)
        tmp_tb_name = tmp_tb_name.upper()
        tbs, _ = self.get_tables()
        tbs = [x.upper() for x in tbs]
        while tmp_tb_name in tbs:
            tmp_tb_name += '_'
        return tmp_tb_name
    
    def oracle_merge_into(self, tb_tgt, tb_src, cols, idcols,
                          rep_keep='src'):
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
        self.execute_sql(sql)
    
    def df_to_oracle(self, df, tb_name, act_type='replace',
                     idcols=None, col_types={}, **kwargs_cols):
        '''
        df存入Oracle
        
        Examples
        --------
        >>> db = SQLAlchemy(dialect='oracle', driver='cx_oracle',
        ...                 username='test', password='xxxxxxxxxxx',
        ...                 host='localhost', port=1521,
        ...                 orcl_pdb=True, database='orclpdb')
        >>> tb_name = 'test3'
        >>> # idcols = None
        >>> idcols = ['code', 'year']
        >>> df = pd.DataFrame({'code': ['001', '002', '003', '004', '005', '006'],
        ...                    'year': ['2011', '2012', '2013', '2014', '2015', '2016'],
        ...                    'value': [1, 2, 3, 4, 5, 6],
        ...                    # 'value1': [1, 2, '3', 4, 5, 6],
        ...                    'value0': ['1a', '2b', '3c', '4d', '5e', '6f']})
        >>> df1 = pd.DataFrame({'code': ['006', '007', '008', '009'],
        ...                     'year': ['2016', '2017', '2018', '2019'],
        ...                     'value': [66, 7, 8, 9],
        ...                     'VALUE2': [10, 11, 12, 13],
        ...                     # 'VALUE3': ['10a', '11b', '12', '13']
        ...                    })
        >>> db.df_to_sql(df, tb_name, act_type='new', idcols=idcols)
        >>> db.df_to_sql(df1, tb_name, act_type='replace', idcols=idcols)
        '''
        df = df.rename(columns={x: x.upper() for x in df.columns})
        assert act_type in ['ignore_tb', 'new', 'insert', 'insert_ignore',
                            'replace', 'insert_ignore_newcols']
        if act_type == 'ignore_tb' and self.has_table(tb_name):
            return
        
        if not act_type in ['new', 'insert']:
            assert not isnull(idcols)
        _, dtype, _, cols_info_dict = get_cols_info_df_oracle(df,
              col_types=change_dict_key(col_types, lambda x: x.upper()),
              **kwargs_cols)
        tbs, _ = self.get_tables()
        tb_not_exists = not tb_name.upper() in [x.upper() for x in tbs]
        if act_type == 'new' or tb_not_exists:
            return df.to_sql(tb_name.lower(), self.conn,
                             if_exists='replace', index=None,
                             dtype=dtype)
        tb_name_tmp = self._get_tmp_tb_name(tb_name+'_TMP')
        df.to_sql(tb_name_tmp.lower(), # 名称大写报错，不知为何
                  self.conn, if_exists='replace', index=None,
                  dtype=dtype)
        try:
            cols = list(df.columns)
            # 表中字段
            fields, _, _ = self.get_fields(tb_name)
            fields = [x.upper() for x in fields]
            # 缺失字段新增
            cols_loss = [x for x in cols if x not in fields]
            for col in cols_loss:
                self.execute_sql('ALTER TABLE {} ADD {}'.format(
                                 tb_name, cols_info_dict[col][0]))
            if act_type == 'insert':
                try:
                    df.to_sql(tb_name.lower(), self.conn,
                              if_exists='append', index=None,
                              dtype=dtype)
                finally:
                    self.drop_table(tb_name_tmp, purge=True)
                return
            idcols = [x.upper() for x in idcols]
            if act_type == 'replace':
                # 有则更新，无则插入
                self.oracle_merge_into(tb_name, tb_name_tmp,
                                       cols, idcols,
                                       rep_keep='src')
            elif act_type == 'insert_ignore':
                self.oracle_merge_into(tb_name, tb_name_tmp,
                                       cols, idcols,
                                       rep_keep='tgt')
            else:
                # 先处理已存在字段
                oldcols = [x for x in cols if x in fields]
                oldcols = list(set(idcols + oldcols))
                if len(oldcols) > 0:
                    self.oracle_merge_into(tb_name, tb_name_tmp,
                                           oldcols, idcols,
                                           rep_keep='tgt')                
                # 再处理新增字段
                if len(cols_loss) > 0:
                    newcols = list(set(idcols + cols_loss))
                    self.oracle_merge_into(tb_name, tb_name_tmp,
                                           newcols, idcols,
                                           rep_keep='src')        
        finally:
            self.drop_table(tb_name_tmp, purge=True)
    
    
def get_conn(dialect='mysql', driver='pymysql', username='root',
             password=None, host='localhost', port=3306,
             database=None, orcl_pdb=False,
             **kwargs):
    '''
    TODO: oracle连接参数有service_name的处理？
    '''
    if dialect.lower() == 'sqlite':
        assert 'dbfile' in kwargs, '必须通过kwargs指定sqlite数据库文件路径'
        file, kwargs_ = get_update_kwargs('dbfile', None, kwargs, func_update=False)
        engine_str = 'sqlite:///{}'.format(file)
        return create_engine(engine_str, **kwargs_)
    if isnull(driver):
        part1 = '{}://'.format(dialect)
    else:
        part1 = '{}+{}://'.format(dialect, driver)
    part2 = '{}:{}@'.format(username, password)
    if not orcl_pdb:
        if dialect == 'oracle':
            assert not isnull(database)
        if isnull(database):
            part3 = '{}:{}'.format(host, port)
        else:
            part3 = '{}:{}/{}'.format(host, port, database)
    else:
        assert not isnull(database)
        part3 = database
    engine_str = part1 + part2 + part3
    engine = create_engine(engine_str, **kwargs)
    return engine

def get_cols_info_df_oracle(df, cols=None, col_types={}, 
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
        指定列类型，如{'col1': 'NVARCHAR(20)', 'col2': 'INT'}，指定的列不做判断，直接返回
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
                char = col + ' NVARCHAR(255)'
        elif 'int' in str(types[col]):
            char = col + ' INT'
        elif 'float' in str(types[col]):
            char = col + ' FLOAT'
        elif 'object' in str(types[col]):
            if col in big_text_cols:
                char = col + ' CLOB'
            else:
                char = col + ' NVARCHAR(255)'
        elif 'datetime' in str(types[col]):
            char = col + ' DATE'
        else:
            raise ValueError('未识别（暂不支持）的字段类型: %s！'%col)
        char_ = ':%s'%(k+1)
        cols_info.append(char)
        placeholder.append(char_)
        cols_info_dict[col] = (char, char_)
    cols_info, placeholder = ', '.join(cols_info), ', '.join(placeholder)
    dtype = {k: eval(v[0].split(' ')[-1]) for k, v in cols_info_dict.items()}
    
    return cols_info, dtype, placeholder, cols_info_dict
        
#%%
if __name__ == '__main__':
    #%%
    db = SQLAlchemy(password='xxxxxxxxxxx', database='test')
    # # 用下面这两句即连接时不选数据库，连接之后再USE，df.to_sql会报错，为啥？
    # db = SQLAlchemy(password='xxxxxxxxxxx', database=None)
    # db.execute_sql('USE test;')
    print(db.execute_sql('show index from test1;'))
    df = pd.DataFrame({'code': ['001', '002', '003'],
                       'year': ['2011', '2012', '2013'],
                       'value': [1, 2, 3],
                       'value1': [4, 5, 6],
                       'value2': [7, 8, 9]})
    df.to_sql('test1', db.conn, if_exists='replace',
              index=None)
    print(db.execute_sql('select * from test1;'))
    print(db.execute_sql('TRUNCATE TABLE test1;'))
    print(db.execute_sql('select * from test1;'))
    
    #%%
    dbo1 = SQLAlchemy(dialect='oracle',
                      driver='cx_oracle',
                      username='c##test', # system
                      password='xxxxxxxxxxx', # sys密码
                      host='localhost',
                      port=1521,
                      database='orcl')
    df = pd.DataFrame({'code': ['001', '002', '003'],
                       'year': ['2011', '2012', '2013'],
                       'value': [1, 2, 3]})
    df.to_sql('test1', dbo1.conn, if_exists='replace',
              index=None)
    print(dbo1.execute_sql('select * from test1'))
    dbo1.execute_sql("insert into test1 values ('b', '2012', 3)")
    print(dbo1.execute_sql('select * from test1'))
    print(dbo1.execute_sql('TRUNCATE TABLE test1'))
    print(dbo1.execute_sql('select * from test1'))
    dbo1.execute_sql("insert into test1 values ('b', '2012', 3)")
    
    #%%
    dbo2 = SQLAlchemy(dialect='oracle',
                      driver='cx_oracle',
                      username='test',
                      password='xxxxxxxxxxx',
                      host='localhost',
                      port=1521,
                      orcl_pdb=True,
                      database='orclpdb')
    df = pd.DataFrame({'code': ['001', '002', '003'],
                       'year': ['2011', '2012', '2013'],
                       'value': ['1', '2', '3']})
    df.to_sql('test1', dbo2.conn, if_exists='replace',
              index=None)
    print(dbo2.execute_sql('select * from test1'))
    dbo2.execute_sql("insert into test1 values ('b', '2012', '3')")
    print(dbo2.execute_sql('select * from test1'))
    print(dbo2.execute_sql('TRUNCATE TABLE test1'))
    print(dbo2.execute_sql('select * from test1'))
    dbo2.execute_sql("insert into test1 values ('b', '2012', 3)")











