# -*- coding: utf-8 -*-
# 文波问题：
# 给定N个整数，找出有多少对数的和为100
# eg. input [2, 98, 2] output 1
# eg. input [1, 99, 98, 2, 2] output 3

if __name__ == '__main__':
    k = 7
    # alist = [1, 99, 1, 99, 98, 2, 2, 98, 3]
    alist = [1, 99, 98, 2, 2]
    v_n = {}
    for v in alist:
        if v not in v_n:
            v_n[v] = 1
            v_n[100-v] = 0
        else:
            v_n[v] += 1
    N = 0
    for v in v_n:
        N += (v_n[v] * v_n[100-v]) / 2
