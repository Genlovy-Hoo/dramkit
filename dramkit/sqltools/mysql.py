# -*- coding: utf-8 -*-


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
    cols_info, cols_type = [], []
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
    cols_info, cols_type = ','.join(cols_info), ','.join(cols_type)

    return cols_info, cols_type


def df_to_sql_pymysql(df, conn, db_name, tb_name, loc='new',
                      cols=None):
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
    loc : str
        | 存入方式：
        |     若为'new'，则新建表（若原表已存在，则会删除重建）
        |     若为'tail'，则插入表的尾部
        |     若为'update'，则将已存在的数据更新
        |     若为'head'，则插入表的前部
    
    References
    -----------    
    https://blog.csdn.net/tonydz0523/article/details/82529941
    '''

    if not cols is None:
        df = df.reindex(columns=cols)
    else:
        cols = df.columns.tolist()

    cur = conn.cursor()

    # 若数据库不存在，则新建
    cur.execute('CREATE DATABASE IF NOT EXISTS {};'.format(db_name))
    # cur.execute('USE {};'.format(db_name))
    conn.select_db(db_name)

    cols_info, cols_type = _get_cols_info_df(df, cols=cols)

    # 表不存在或强制建新表
    if loc == 'new':
        # 创建table
        cur.execute('DROP TABLE IF EXISTS {};'.format(tb_name))
        cur.execute('CREATE TABLE {a}({b});'.format(a=tb_name, b=cols_info))
    cur.execute('CREATE TABLE IF NOT EXISTS {a}({b});'.format(
                                                    a=tb_name, b=cols_info))
    # TODO
    # 检查字段是否已经存在，不存在新建

    # 数据更新
    if loc in ['new', 'tail']:
        # 批量插入新数据
        values = df.values.tolist()
        values = [tuple(x) for x in values]
        cur.executemany('INSERT INTO {a} ({b}) VALUES ({c});'.format(
                        a=tb_name, b=','.join(cols), c=cols_type),
                        values)
    elif loc == 'update':
        pass
    elif loc == 'head':
        pass

    cur.close()
    conn.commit()
