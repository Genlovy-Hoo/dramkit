# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm


def app(S, T, sigma, q):
    '''
    平均价格亚式期权模型（“AAP模型”）
    '''
    sigma2 = np.power(sigma, 2)
    p1 = sigma2 * T
    ep1 = np.exp(p1)
    p2 = np.log(2 * (ep1 - p1 - 1))
    p3 = 2 * np.log(ep1 - 1)
    vT = np.power(p1 + p2 - p3, 1/2)
    P = S * np.exp(-1 * q * T) * (norm.cdf(vT / 2) - norm.cdf(-1 * vT / 2))
    return P


def cal_limit_stock_value(S, T, sigma, q):
    '''
    | S：估值日在证券交易所上市交易的同一股票的公允价值
    | T：剩余限售期，以年为单位表示
    | sigma：股票在剩余限售期内的股价的预期年化波动率
    | q：股票预期年化股利收益率
    | 
    | 参考：
    | 中证指数有限公司流通受限股票流动性折扣计算说明v2.docx
    | https://www.sohu.com/a/193630728_656666
    '''
    P = app(S, T, sigma, q)
    LoMD = P / S
    FV = S * (1 - LoMD)
    return FV, (P, LoMD)


if __name__ == '__main__':
    S = 10 # 估值日在证券交易所上市交易的同一股票的公允价值
    T = 1 # 剩余限售期，以年为单位表示
    sigma = 0.5 # 股票在剩余限售期内的股价的预期年化波动率
    q = 5 / 100 # 股票预期年化股利收益率

    FV, _ = cal_limit_stock_value(S, T, sigma, q)
    print('P: {}'.format(_[0]))
    print('LoMD: {}'.format(_[1]))
    print('FV: {}'.format(FV))
