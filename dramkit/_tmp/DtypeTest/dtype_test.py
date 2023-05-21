# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dramkit import TimeRecoder
from dramkit.iotools import (get_file_info,
                             load_df_pd,
                             df2file_pd,
                             make_dir,
                             del_dir)

testdir = './dtype_test/'
make_dir(testdir)


def gen_data():
    np.random.seed = 5262
    # df_size = 10_000_000
    df_size = 1000000
    df = pd.DataFrame({
        'a': ['你好'] * df_size,
        'b': ['我好'] * df_size,
        'c': ['她好'] * df_size,
        'd': ['狗好'] * df_size,
        'e': ['猫不好'] * df_size
    })
    return df


# def gen_data():
#     np.random.seed = 5262
#     df_size = 10_000_000
#     df_size = 1000000
#     df = pd.DataFrame({
#         'a': np.random.rand(df_size),
#         'b': np.random.rand(df_size),
#         'c': np.random.rand(df_size),
#         'd': np.random.rand(df_size),
#         'e': np.random.rand(df_size)
#     })
#     return df


def test_save_read(df, dtype):
    fpath = testdir + 'dtype_test.%s'%dtype
    trw = TimeRecoder()
    if dtype not in 'hdf':
        # exec('df.to_{}("{}")'.format(dtype, fpath))
        df2file_pd(df, fpath)
    else:
        # exec('df.to_{}("{}", key="df")'.format(dtype, fpath))
        df2file_pd(df, fpath, key='df')
    tuw = trw.used()
    fsize = get_file_info(fpath)['size_mb']
    # tmp = None
    trr = TimeRecoder()
    # exec('tmp = pd.read_{}("{}")'.format(dtype, fpath))
    exec('tmp = load_df_pd(fpath)')
    tur = trr.used()
    return {'dtype': dtype, 'twrite': tuw, 'tread': tur,
            'fsize': fsize}


if __name__ == '__main__':
    df = gen_data()
    
    dtypes = ['csv', 'feather', 'pickle', 'hdf']
    
    res = []
    for dtype in tqdm(dtypes):
        re = test_save_read(df, dtype)
        res.append(re)
    res = pd.DataFrame(res)
    
    
    del_dir(testdir)
    
    
    
    
