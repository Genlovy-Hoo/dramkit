# -*- coding: utf-8 -*-

# https://blog.csdn.net/youcans/article/details/116448853

# LinearRegression_v2.py
# Linear Regression with statsmodels (OLS: Ordinary Least Squares)
# v2.0: 调用 statsmodels 实现多元线性回归
# 日期：2021-05-04

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# 主程序
def main():  # 主程序

    # 生成测试数据:
    nSample = 100
    x0 = np.ones(nSample)  # 截距列 x0=[1,...1]
    x1 = np.linspace(0, 20, nSample)  # 起点为 0，终点为 10，均分为 nSample个点
    x2 = np.sin(x1)
    x3 = (x1-5)**2
    X = np.column_stack((x0, x1, x2, x3))  # (nSample,4): [x0,x1,x2,...,xm]
    beta = [5., 0.5, 0.5, -0.02] # beta = [b1,b2,...,bm]
    yTrue = np.dot(X, beta)  # 向量点积 y = b1*x1 + ...+ bm*xm
    yTest = yTrue + 0.5 * np.random.normal(size=nSample)  # 产生模型数据

    # 多元线性回归：最小二乘法(OLS)
    model = sm.OLS(yTest, X)  # 建立 OLS 模型: Y = b0 + b1*X + ... + bm*Xm + e
    results = model.fit()  # 返回模型拟合结果
    yFit = results.fittedvalues  # 模型拟合的 y值
    print(results.summary())  # 输出回归分析的摘要
    print("\nOLS model: Y = b0 + b1*X + ... + bm*Xm")
    print('Parameters: ', results.params)  # 输出：拟合模型的系数

    # 绘图：原始数据点，拟合曲线，置信区间
    prstd, ivLow, ivUp = wls_prediction_std(results) # 返回标准偏差和置信区间
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, yTest, 'o', label="data")  # 实验数据（原始数据+误差）
    ax.plot(x1, yTrue, 'b-', label="True")  # 原始数据
    ax.plot(x1, yFit, 'r-', label="OLS")  # 拟合数据
    ax.plot(x1, ivUp, '--',color='orange', label="ConfInt")  # 置信区间 上届
    ax.plot(x1, ivLow, '--',color='orange')  # 置信区间 下届
    ax.legend(loc='best')  # 显示图例
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return

#= 关注 Youcans，分享原创系列 https://blog.csdn.net/youcans =
if __name__ == '__main__':
    main()
