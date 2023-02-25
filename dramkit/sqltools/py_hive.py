# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from copy import copy
from pyhive import hive
from impala import dbapi
from dramkit.gentools import (isnull,
                              check_list_arg,
                              df_na2value,
                              change_dict_key,
                              print_used_time)
from dramkit.sqltools.py_mysql import (get_cols_info_df,
                                       df_to_sql_insert_values)


class PyHive(object):
    def __init__(self, host, username, password=None,
                 database=None, port=10000,
                 driver='pyhive', **kwargs):
        self.driver = driver
        self.__db_conn_args = {'host': host,
                               'user': username,
                               'password': password,
                               'database': database,
                               'port': port
                               }
        self.__db_conn_args.update(kwargs)
        self.conn = self.get_conn(**self.__db_conn_args)
        self.hive_version = self.get_version()
        
    def get_conn(self, *args, **kwargs):
        # print(locals())
        if self.driver == 'pyhive':
            return get_conn_pyhive(*args, **kwargs)
        elif self.driver == 'impala':
            return get_conn_impala(*args, **kwargs)
        
    def copy(self):
        res = copy(self)
        res.conn = self.get_conn(**self.__db_conn_args)
        return res

    def execute_sql(self, sql_str, db_name=None, to_df=True):
        res = execute_sql(conn=self.conn, sql_str=sql_str,
                          db_name=db_name, to_df=to_df)
        return res
    
    def get_version(self):
        return self.execute_sql('SELECT VERSION()', to_df=False)[0][0][:3]
    
    def now_database(self):
        res = self.execute_sql('SELECT CURRENT_DATABASE()', to_df=False)
        return res[0][0]
    
    def show_tables(self, db_name=None):
        res = self.execute_sql('SHOW TABLES',
                               db_name=db_name, to_df=False)
        return [x[0] for x in res]
    
    def has_table(self, tb_name, db_name=None):
        tbs_ = [x.upper() for x in self.show_tables(db_name=db_name)]
        return tb_name.upper() in tbs_
    
    def drop_table(self, tb_name, db_name=None, purge=True):
        '''删除表'''
        sql_str = 'DROP TABLE IF EXISTS {}'.format(tb_name)
        if purge:
            sql_str = 'DROP TABLE IF EXISTS {} PURGE'.format(tb_name)
        self.execute_sql(sql_str, db_name=db_name, to_df=False)
        
    def get_fields(self, tb_name, db_name=None):
        '''获取表字段名列表'''
        sql_str = 'DESC {}'.format(tb_name)
        fields_info = self.execute_sql(sql_str,
                           db_name=db_name, to_df=True)
        fields = fields_info['col_name'].tolist()
        info_dict = fields_info.set_index('col_name')['data_type'].to_dict()
        return fields, fields_info, info_dict
    
    def drop_cols(self, tb_name, cols, db_name=None):
        '''
        删除列
        
        Notes
        -----
        | Hive中，如果表是orc格式的，不支持删除，会报错，
        | 如果是textfile格式，则可以删除。
        '''
        cols = check_list_arg(cols)
        allcols, cols_info, _ = self.get_fields(tb_name, db_name=db_name)
        keeps = [x.upper() for x in allcols if x.upper() not in [y.upper() for y in cols]]
        cols_info['col_name'] = cols_info['col_name'].apply(lambda x: x.upper())
        keeps_info = cols_info.set_index('col_name')['data_type'].to_dict()
        keeps_info = ['%s %s'%(k, v) for k, v in keeps_info.items() if k in keeps]
        sql = 'ALTER TABLE %s REPLACE COLUMNS (%s)'%(tb_name, ', '.join(keeps_info))
        self.execute_sql(sql, db_name=db_name, to_df=False)
        
    def _get_tmp_tb_name(self, tmp_tb_name):
        assert isinstance(tmp_tb_name, str)
        tmp_tb_name = tmp_tb_name.upper()
        tbs = self.show_tables()
        tbs = [x.upper() for x in tbs]
        while tmp_tb_name in tbs:
            tmp_tb_name += '_'
        return tmp_tb_name
    
    def create_table(self, tb_name, cols_info, idcols=None,
                     db_name=None, force=False):
        '''
        新建表
        
        Examples
        --------
        >>> db = PyHive(host='192.168.118.128', port=10000,
        ...             username='root', database='test')
        >>> idcols = None
        >>> idcols = ['a', 'c']
        >>> db.create_table('test15',
        ...                 ('a VARCHAR(255)',
        ...                  'b FLOAT NOT NULL',
        ...                  'c DATE'),
        ...                 idcols=idcols, db_name='test1',
        ...                 force=True)
        '''
        has_ = self.has_table(tb_name, db_name=db_name)
        if has_ and not force:
            return
        if has_ and force:
            self.execute_sql('DROP TABLE %s PURGE'%tb_name, db_name=db_name)
        idcols = check_list_arg(idcols, allow_none=True)
        colstr = '('
        colstr = colstr + ', '.join(cols_info)
        if not isnull(idcols):
            if self.hive_version >= '3.0':
                pkstr = '\nCONSTRAINT uni_{} UNIQUE ({}) DISABLE NOVALIDATE'.format(tb_name, ', '.join(idcols))
            else:
                pkstr = '\nPRIMARY KEY ({}) DISABLE NOVALIDATE'.format(', '.join(idcols))
            colstr = colstr + ',' + pkstr + ')'
        else:
            colstr = colstr + ')'
        if not isnull(idcols):
            sql = '''CREATE TABLE {} 
                     {} 
                     CLUSTERED BY ({}) INTO 31 BUCKETS 
                     STORED AS ORC TBLPROPERTIES('transactional'='true')
                  '''.format(tb_name, colstr, ', '.join(idcols))
        else:
            sql = '''CREATE TABLE {} 
                     {} 
                     STORED AS ORC TBLPROPERTIES('transactional'='true')
                  '''.format(tb_name, colstr)
        self.execute_sql(sql, db_name=db_name)
        
    def merge_into(self, *args, **kwargs):
        if self.hive_version >= '2.2':
            return self.merge_into1(*args, **kwargs)
            # try:
            #     return self.merge_into1(*args, **kwargs)
            # except:
            #     return self.merge_into2(*args, **kwargs)
            #     # raise
        else:
            return self.merge_into2(*args, **kwargs)
    
    def merge_into1(self, tb_tgt, tb_src, replace_cols, idcols,
                    rep_keep='src', db_name=None):
        assert rep_keep in ['src', 'tgt']
        assert isinstance(replace_cols, list) and isinstance(idcols, list)
        noidcols = [x for x in replace_cols if not x in idcols]
        on_ = ' AND '.join(['t.%s=s.%s'%(c, c) for c in idcols])
        update_ = ''
        if rep_keep == 'src':
            update_ = '''WHEN MATCHED THEN 
                         UPDATE SET {} 
                      '''.format(', '.join(['%s=s.%s'%(c, c) for c in noidcols]))
        t_cols = self.get_fields(tb_tgt, db_name=db_name)[0]
        insert_ = 'INSERT VALUES ({})'.format(', '.join(['s.%s'%c for c in t_cols]))
        sql = '''MERGE INTO {} AS t 
                 USING {} AS s 
                 ON ({}) 
                 {}
                 WHEN NOT MATCHED THEN
                 {}
              '''.format(tb_tgt, tb_src, on_, update_, insert_)
        self.execute_sql('SET hive.auto.convert.join=false')
        self.execute_sql(sql, db_name=db_name)
        
    def merge_into2(self, tb_tgt, tb_src, replace_cols, idcols,
                    rep_keep='src', db_name=None):
        assert rep_keep in ['src', 'tgt']
        assert isinstance(replace_cols, list) and isinstance(idcols, list)
        if rep_keep == 'tgt':
            tb_tgt_, tb_src_ = tb_src, tb_tgt
        else:
            tb_tgt_, tb_src_ = tb_tgt, tb_src
        idcols = [x.lower() for x in idcols]
        replace_cols = [x.lower() for x in replace_cols]
        on_ = ' AND '.join(['t.%s=s.%s'%(c, c) for c in idcols])
        t_cols = self.get_fields(tb_tgt, db_name=db_name)[0]
        t_cols = [x.lower() for x in t_cols]
        select_ = []
        for col in t_cols:
            if col in replace_cols:
                select_.append('COALESCE(s.{c}, t.{c}) AS {c}'.format(c=col))
            else:
                select_.append('t.{c} AS {c}'.format(c=col))
        select_ = ', \n'.join(select_)
        sql = '''INSERT OVERWRITE TABLE {f1} 
                 SELECT {f4} 
                 FROM {f2} s 
                 FULL OUTER JOIN {f5} t 
                 ON {f3} 
              '''.format(f1=tb_tgt, f2=tb_src_, f3=on_, f4=select_, f5=tb_tgt_)

        self.execute_sql('SET hive.auto.convert.join=false')
        self.execute_sql('SET hive.stats.autogather=false')
		# self.execute_sql(sql, db_name=db_name)
        sql = '''INSERT INTO TABLE {f1} 
                 SELECT {f4} 
                 FROM {f2} s 
                 FULL OUTER JOIN {f5} t 
                 ON {f3} 
              '''.format(f1=tb_tgt, f2=tb_src_, f3=on_, f4=select_, f5=tb_tgt_)
        self.execute_sql('TRUNCATE TABLE {}'.format(tb_tgt))
        self.execute_sql(sql, db_name=db_name)

    
    @print_used_time
    def df_to_sql(self, df, tb_name, act_type='replace',
                  db_name=None, cols=None, idcols=None,
                  col_types={}, na_val=None, 
                  inf_val='inf', _inf_val='-inf',
                  **kwargs_cols):
        '''
        把pandas.DataFrame存入Hive数据库中
        
        Examples
        --------
        >>> db = PyHive(host='192.168.118.128', port=10000,
        ...             username='root', database='test')
        >>> db = PyHive(host='192.168.118.128', port=10000,
        ...             username='root', database='test',
        ...             password='123456', driver='impala',
        ...             auth_mechanism='PLAIN')
        >>> tb_name = 'test'
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
        >>> db.df_to_sql(df, tb_name, act_type='new', idcols=idcols)
        >>> db.df_to_sql(df1, tb_name, act_type='insert_ignore_newcols', idcols=idcols)
        '''
        
        assert act_type in ['ignore_tb', 'new', 'insert', 'insert_ignore',
                            'replace', 'insert_ignore_newcols']
        if act_type == 'ignore_tb' and self.has_table(tb_name, db_name=db_name):
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

        cur_ = self.conn.cursor()
        
        # 若数据库不存在，则新建
        if not isnull(db_name):
            cur_.execute('CREATE DATABASE IF NOT EXISTS {}'.format(db_name))
            cur_.execute('USE {}'.format(db_name))

        cols_info, dtype, ph, cols_info_dict = get_cols_info_df(
            df, cols=cols, col_types=change_dict_key(col_types, lambda x: x.upper()),
            **kwargs_cols)

        # 表不存在则新建
        idcols = check_list_arg(idcols, allow_none=True)
        idcols = [x.upper() for x in idcols] if not isnull(idcols) else idcols
        force = True if act_type == 'new' else False
        self.create_table(tb_name, cols_info.split(', '),
                          db_name=db_name, idcols=idcols,
                          force=force)
        
        # 表字段列表
        fields, _, _ = self.get_fields(tb_name, db_name=db_name)
        fields = [x.upper() for x in fields]

        # 检查字段是否已经存在，不存在新建
        cols_loss = [x for x in cols if x not in fields]
        for col in cols_loss:
            cur_.execute('ALTER TABLE {} ADD COLUMNS ({})'.format(
                 tb_name, cols_info_dict[col][0]))
            
        # 全部字段
        fields_, _, info_dict = self.get_fields(tb_name, db_name=db_name)
        fields_ = [x.upper() for x in fields_]
        
        # 数据更新
        # values = df.values.tolist()
        # TODO: pyhive用cur_.executemany报错，原因待查
        # impala用cur_.executemany很慢
        values_str = df_to_sql_insert_values(df, cols=cols)
        if act_type == 'new' or isnull(idcols):
            # TODO: 低版本不能全字段插入，待确定从哪个版本开始
            # TODO: 解决低版本不能全字段插入式只更新部分字段的问题
            if self.hive_version >= '3.0':
                # cur_.executemany('INSERT INTO {a} ({b}) VALUES ({c})'.format(
                #                  a=tb_name, b=','.join(cols), c=ph),
                #                  values)
                cur_.execute('INSERT INTO {a} ({b}) VALUES {c}'.format(
                             a=tb_name, b=','.join(cols), c=values_str))
            else:
                # 解决报错：This command is not allowed on an ACID table xxx.xxx_table  with a non-ACID transaction manager
                cur_.execute('SET hive.support.concurrency=true')
                cur_.execute('SET hive.txn.manager=org.apache.hadoop.hive.ql.lockmgr.DbTxnManager')
                cur_.execute('INSERT INTO {a} VALUES {c}'.format(
                             a=tb_name, c=values_str))
        elif act_type == 'insert':
            # 批量插入新数据
            try:
                # TODO: 低版本不能全字段插入，待确定从哪个版本开始
                # TODO: 解决低版本不能全字段插入式只更新部分字段的问题
                if self.hive_version >= '3.0':
                    # cur_.executemany('INSERT INTO {a} ({b}) VALUES ({c})'.format(
                    #                  a=tb_name, b=','.join(cols), c=ph),
                    #                  values)
                    cur_.execute('INSERT INTO {a} ({b}) VALUES {c}'.format(
                                  a=tb_name, b=','.join(cols), c=values_str))
                else:
                    # 解决报错：This command is not allowed on an ACID table xxx.xxx_table  with a non-ACID transaction manager
                    cur_.execute('SET hive.support.concurrency=true')
                    cur_.execute('SET hive.txn.manager=org.apache.hadoop.hive.ql.lockmgr.DbTxnManager')
                    cur_.execute('INSERT INTO {a} VALUES {c}'.format(
                                 a=tb_name, c=values_str))
            except:
                self.drop_cols(tb_name, cols_loss, db_name=db_name)
                raise
        else:
            tb_name_tmp = self._get_tmp_tb_name(tb_name+'_TMP')
            df_tmp = df.reindex(columns=fields_)
            self.df_to_sql(df_tmp, tb_name_tmp, act_type='new',
                           db_name=db_name, cols=None, idcols=idcols,
                           col_types=change_dict_key(info_dict, lambda x: x.upper()),
                           na_val=na_val, inf_val=inf_val,
                           _inf_val=_inf_val, **kwargs_cols)
            try:
                # time.sleep(2)
                if act_type == 'replace':
                    # 有则更新，无则插入
                    self.merge_into(tb_name, tb_name_tmp, cols, idcols,
                                    rep_keep='src', db_name=db_name)
                elif act_type == 'insert_ignore':
                    # 若有重复，保留已存在的不更新
                    self.merge_into(tb_name, tb_name_tmp, cols, idcols,
                                    rep_keep='tgt', db_name=db_name)
                else:
                    # 先处理已存在字段
                    oldcols = [x for x in cols if x in fields]
                    oldcols = list(set(idcols + oldcols))
                    if len(oldcols) > 0:
                        self.merge_into(tb_name, tb_name_tmp, oldcols, idcols,
                                        rep_keep='tgt', db_name=db_name)
                    # 再处理新增字段
                    if len(cols_loss) > 0:
                        newcols = list(set(idcols + cols_loss))
                        self.merge_into(tb_name, tb_name_tmp, newcols, idcols,
                                        rep_keep='src', db_name=db_name)
            finally:
                self.drop_table(tb_name_tmp, db_name=db_name, purge=True)
        cur_.close()
        self.conn.commit()
        
        
def get_conn_pyhive(host, user, password=None,
                    database='default', port=10000,
                    auth=None, **kwargs):
    '''连接数据库'''
    conn = hive.Connection(host=host, username=user,
                           password=password, port=port,
                           database=database, auth=auth,
                           **kwargs)
    return conn


def get_conn_impala(host, user, password=None, port=10000,
                    database='default', auth_mechanism=None,
                    **kwargs):
    conn = dbapi.connect(host=host, user=user, port=port,
                         auth_mechanism=auth_mechanism,
                         password=password, database=database,
                         **kwargs)
    return conn


def execute_sql(conn, sql_str, db_name=None, to_df=True):
    '''执行sql语句并返回结果'''
    cur = conn.cursor()
    if not isnull(db_name):
        cur.execute('USE {}'.format(db_name))
    cur.execute(sql_str)
    if isnull(cur.description):
        cur.close()
        conn.commit()
        return None
    res = cur.fetchall()
    if to_df:
        cols = [x[0] for x in cur.description]
        res = pd.DataFrame(res, columns=cols)
        res = res.rename(columns={x: x.split('.')[-1] for x in res.columns})
    cur.close()
    conn.commit()
    return res


if __name__ == '__main__':
    '''
    # 方式1：无密码，正常
    db1 = PyHive(host='192.168.118.128', port=10000,
                 username='root', database='test')
    print(db1.execute_sql('show databases'))
    '''
    '''
    # 方式2：'LDAP'+密码，正常
    db2 = PyHive(host='192.168.118.128', port=10000,
                 username='root', password='123456',
                 database='test', auth='LDAP')
    print(db2.execute_sql('show databases'))
    '''
    '''
    # 方式3：'CUSTOM'+密码，正常
    db3 = PyHive(host='192.168.118.128', port=10000,
                 username='root', password='123456',
                 database='test', auth='CUSTOM')
    print(db3.execute_sql('show databases'))
    '''
    '''
    # 方式4：'KERBEROS'，报错
    db4 = PyHive(host='192.168.118.128', port=10000,
                 username='root', database='test',
                 auth='KERBEROS',
                 kerberos_service_name='hive'
                 )
    print(db4.execute_sql('show databases'))
    '''
    
    '''
    # 方式5：impala，'PLAIN'+密码，正常
    db5 = PyHive(host='192.168.118.128', port=10000,
                 username='root', database='test',
                 password='123456', auth_mechanism='PLAIN',
                 driver='impala')
    print(db5.execute_sql('show databases'))
    '''
    '''
    # 方式6：impala，'LDAP'+密码，正常
    db6 = PyHive(host='192.168.118.128', port=10000,
                 username='root', database='test',
                 password='123456', auth_mechanism='LDAP',
                 driver='impala')
    print(db6.execute_sql('show databases'))
    '''
    '''
    # 方式7：impala，'GSSAPI'+密码，报错
    db7 = PyHive(host='192.168.118.128', port=10000,
                 username='root', database='test',
                 password='123456', auth_mechanism='GSSAPI',
                 driver='impala')
    print(db7.execute_sql('show databases'))
    '''     
    
    db = PyHive(host='192.168.118.128', port=10000,
                username='root', database='test')
    db.execute_sql('create database if not exists test')
    print(db.execute_sql('show databases'))
    
    # db.create_table(tb_name='test_up',
    #                 cols_info=('code varchar(255)',
    #                            'year int',
    #                            'value1 float',
    #                            'value2 float'),
    #                 idcols=['code', 'year'],
    #                 db_name=None, force=True)
    # db.create_table(tb_name='test_up2',
    #                 cols_info=('code varchar(255)',
    #                            'year int',
    #                            'value1 float',
    #                            'value2 float'),
    #                 idcols=['code', 'year'],
    #                 db_name=None, force=True)
    # db.execute_sql('''INSERT INTO test_up VALUES 
    #                   ('a', 2011, 1.0, 1.1),
    #                   ('b', 2012, 2.0, 2.1)
    #                ''')
    # db.execute_sql('''INSERT INTO test_up2 VALUES 
    #                   ('b', 2012, 2.1, 2.1),
    #                   ('c', 2013, 3.1, 3.1)
    #                ''')
    # df1 = db.execute_sql('''SELECT * FROM test_up a 
    #                         FULL OUTER JOIN test_up2 b 
    #                         ON a.code = b.code 
    #                         AND a.year = b.year
    #                      ''')
    # df2 = db.execute_sql(
    #     '''SELECT b.code, b.year,
    #               COALESCE(b.value1, a.value1) as value1,
    #               COALESCE(b.value2, a.value2) as value2 
    #               FROM test_up2 b 
    #               FULL OUTER JOIN test_up a 
    #               ON b.code = a.code AND b.year = a.year
    #     '''
    #     )
    # df3 = db.execute_sql(
    #     '''SELECT COALESCE(b.code, a.code) as code,
    #               COALESCE(b.year, a.year) as year,
    #               COALESCE(b.value1, a.value1) as value1,
    #               COALESCE(b.value2, a.value2) as value2 
    #               FROM test_up2 b 
    #               FULL OUTER JOIN test_up a 
    #               ON a.code = b.code AND a.year = b.year
    #     '''
    #     )
    # df4 = db.execute_sql(
    #     '''INSERT OVERWRITE TABLE test_up
    #        SELECT COALESCE(b.code, a.code) as code,
    #               COALESCE(b.year, a.year) as year,
    #               COALESCE(b.value1, a.value1) as value1,
    #               COALESCE(b.value2, a.value2) as value2 
    #               FROM test_up2 b 
    #               FULL OUTER JOIN test_up a 
    #               ON b.code = a.code AND b.year = a.year
    #     '''
    #     )
    # db.merge_into2('test_up', 'test_up2',
    #                ['code', 'year', 'value1', 'value2'],
    #                ['code', 'year'],
    #                rep_keep='tgt')
    
    
    # db.execute_sql('create database if not exists test1')
    # db.execute_sql('create database if not exists test2')
    # db.execute_sql('''create table if not exists test 
    #                   (code varchar(255), year int, value float)
    #                ''', db_name='test')
    # db.execute_sql('''create table if not exists test11 
    #                   (code varchar(255), year int, value float)
    #                ''', db_name='test1')
    # db.execute_sql('''create table if not exists test21 
    #                   (code varchar(255), year int, value float)
    #                ''', db_name='test2')
    # db.execute_sql('''insert into test values ('c', 2013, 3)
    #                ''', db_name='test')
    # db.execute_sql('''insert into test11 values ('d', 2014, 4)
    #                ''', db_name='test1')
    # df = db.execute_sql('select * from test', db_name='test')
    # df1 = db.execute_sql('select * from test11', db_name='test1')

    
    








