# -*- coding: utf-8 -*-

import time
from dramkit.gentools import TimeRecoder
from dramkit.datetimetools import timestamp2str


if __name__ == '__main__':
    ts = time.time() #* 1000
    n = 1000000
    tz = 'utc'
    
    tr1 = TimeRecoder()
    for _ in range(n):
        t1 = timestamp2str(ts, tz=tz, method=1)
    tr1.used()
    
    tr2 = TimeRecoder()
    for _ in range(n):
        t2 = timestamp2str(ts, tz=tz, method=2)
    tr2.used()
    
    tr3 = TimeRecoder()
    for _ in range(n):
        t3 = timestamp2str(ts, tz=tz, method=3)
    tr3.used()
