# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import time
    n = 200
    for k in range(n):
        pstr = '{}/{}, {}%'.format(k+1, n, round(100*(k+1)/n, 2))
        print('\r', pstr, end='', flush=True)
        time.sleep(0.000001)
