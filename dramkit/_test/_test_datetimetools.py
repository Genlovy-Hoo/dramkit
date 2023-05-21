# -*- coding: utf-8 -*-

import time
from dramkit.gentools import func_runtime_test
from dramkit.datetimetools import timestamp2str


if __name__ == '__main__':
    ts = time.time() #* 1000
    n = 200000
    tz = 'utc'
    
    return_all = False
    
    t1, res1 = func_runtime_test(timestamp2str, n=n,
                                 return_all=return_all,
                                 t=ts, tz=tz, method=1)
    t2, res2 = func_runtime_test(timestamp2str, n=n,
                                 return_all=return_all,
                                 t=ts, tz=tz, method=2)
    t3, res3 = func_runtime_test(timestamp2str, n=n,
                                 return_all=return_all,
                                 t=ts, tz=tz, method=3)
