# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def grading(ser,thresh_list):
    ser_copy = ser.copy()
    n = len(thresh_list)
    if (thresh_list == np.sort(thresh_list)).sum() == n:
        ser_copy.loc[ser<thresh_list[0]] = 0
        for i in range(1,n):
            ser_copy.loc[(ser>=thresh_list[i-1])&(ser<thresh_list[i])] = i
        ser_copy[ser>=thresh_list[n-1]] = n
    elif (thresh_list == np.sort(thresh_list)[::-1]).sum()== n:
        ser_copy.loc[ser>thresh_list[0]] = 0
        for i in range(1,n):
            ser_copy.loc[(ser<=thresh_list[i-1])&(ser>thresh_list[i])] = i
        ser_copy[ser<=thresh_list[n-1]] = n
    else:
        print('输入阈值有问题')
        raise
    return ser_copy


if __name__ == '__main__':
    ser = pd.Series([0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89])
    ser1=grading(ser, [0.3, 0.4, 0.5, 0.6, 0.7])
    ser2=grading(ser, [0.3, 0.4, 0.5, 0.6])
    ser3=grading(ser, [0.5, 0.6, 0.7, 0.8])
    
    ser=pd.Series([0.23, 0.34, 0.45, 0.56, 0.67, -0.5])
    df = pd.DataFrame({'ser': ser})
    df['ser1'] = grading(df['ser'],
                         [0.8-1,0.7-1,0.6-1,0.5-1])
    
    
    
    
    
    
    
    
    
    
    
    
    