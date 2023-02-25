# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:03:37 2023

@author: huyy43273
"""

import pandas as pd

def link_lists(lists):
    '''
    | 将多个列表连接成一个列表
    | 注：lists为列表，其每个元素也为一个列表
    
    Examples
    --------
    >>> a = [1, 2, 3]
    >>> b = [4, 5, 6]
    >>> c = ['a', 'b']
    >>> d = [a, b]
    >>> link_lists([a, b, c, d])
    [1, 2, 3, 4, 5, 6, 'a', 'b', [1, 2, 3], [4, 5, 6]]
    '''
    assert isinstance(lists, (list, tuple))
    assert all([isinstance(x, list) for x in lists])
    newlist = []
    for item in lists:
        newlist.extend(item)
    return newlist


def maxnsort(df, n=10, ascending=False):
    idxs = df.index.tolist()
    res = []
    for idx in idxs:
        tmp = df.loc[idx]
        tmp = tmp.sort_values(ascending=ascending)
        tmp = tmp.to_frame().iloc[:n, :].reset_index()
        tmp = tmp.to_dict(orient='split')['data']
        tmp = link_lists(tmp)
        res.append(tmp)
    res = pd.DataFrame(res)
    res.columns = link_lists([['fac%s'%x, 'fac%sshap'%x] for x in range(1, n+1)])
    return res