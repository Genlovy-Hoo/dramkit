# -*- coding: utf-8 -*-

import pandas as pd
from sqlalchemy import create_engine
from dramkit.gentools import isnull


class SQLAlchemy(object):
    
    def __init__(self,
                 dialect='mysql',
                 driver='pymysql',
                 username='root',
                 password=None,
                 host='localhost',
                 port=3306,
                 database=None,
                 **kwargs):
        self.conn = get_conn(dialect=dialect,
                             driver=driver,
                             username=username,
                             password=password,
                             host=host,
                             port=port,
                             database=database,
                             **kwargs)
        self.db_name = database
        self.db_name_ori = database
        
    def execute_sql(self, sql_str, to_df=True):
        with self.conn.connect() as conn:
            res = conn.execute(sql_str)
            if isnull(res.cursor):
                return None
            if to_df:
                cols = [x[0] for x in res.cursor.description]
                return pd.DataFrame(res, columns=cols)
            return res
    
    
def get_conn(dialect='mysql', driver='pymysql', username='root',
             password=None, host='localhost', port=3306,
             database=None, **kwargs):
    if isnull(driver):
        part1 = '{}://'.format(dialect)
    else:
        part1 = '{}+{}://'.format(dialect, driver)
    part2 = '{}:{}@'.format(username, password)
    part3 = '{}:{}/{}'.format(host, port, database)
    engine_str = part1 + part2 + part3
    engine = create_engine(engine_str, **kwargs)
    return engine
        
        
if __name__ == '__main__':
    db = SQLAlchemy(password='xxxxxxxxxxx', database='test')
    print(db.execute_sql('show index from test1;'))
    df = pd.DataFrame({'code': ['001', '002', '003'],
                       'year': ['2011', '2012', '2013'],
                       'value': [1, 2, 3]})
    df.to_sql('test1', db.conn, if_exists='replace', index=None)
    print(db.execute_sql('select * from test1;'))
    print(db.execute_sql('TRUNCATE TABLE test1;'))
    print(db.execute_sql('select * from test1;'))
    











