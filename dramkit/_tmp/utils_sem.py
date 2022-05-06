# -*- coding: utf-8 -*-

# https://zhuanlan.zhihu.com/p/32531377

import time
import numpy as np
import warnings


def cfa1(data, lam, step=0.01, max_iter=5000, rdd=3):
    s, var_e, phi = np.cov(data, rowvar=False, bias=True), np.eye(lam.shape[0]), np.eye(lam.shape[1])
    for i in range(max_iter):
        sigma = np.dot(np.dot(lam, phi), lam.transpose()) + var_e
        sigma_inv = np.linalg.inv(sigma)
        omega = sigma_inv - np.dot(sigma_inv, np.dot(s, sigma_inv))
        omega_lam = np.dot(omega, lam)
        dlam, dphi, dvar_e = 2 * np.dot(omega_lam, phi), np.dot(lam.transpose(), omega_lam), omega
        dlam[lam == 0], dphi[range(lam.shape[1]), range(lam.shape[1])], dvar_e[var_e == 0] = 0, 0, 0
        lam, phi, var_e = lam - step * dlam, phi - step * dphi, var_e - step * dvar_e
    return np.round(lam, rdd), np.round(phi, rdd), np.round(var_e, rdd)


def cfa(data, lam, step=0.01, max_iter=10000, rdd=3, tol=1e-7):
    # 结构方程模型中的测量模型，验证性因子分析参数估计
    # 下面是样本协方差矩阵
    s = np.cov(data, rowvar=False, bias=True)
    # 误差协方差矩阵初值
    var_e = np.eye(lam.shape[0])
    # 潜在变量协方差矩阵初值
    phi = np.eye(lam.shape[1])
    for i in range(max_iter):
        # 估计协方差矩阵
        sigma = np.dot(np.dot(lam, phi), lam.transpose()) + var_e
        sigma_inv = np.linalg.inv(sigma)
        omega = sigma_inv - np.dot(sigma_inv,np.dot(s, sigma_inv))
        omega_lam = np.dot(omega, lam)
        # lambda的梯度
        dlam = 2 * np.dot(omega_lam, phi)
        dlam[lam == 0] = 0
        # phi的梯度
        dphi = np.dot(lam.transpose(), omega_lam)
        dphi[range(lam.shape[1]), range(lam.shape[1])] = 0
        # var_e的梯度
        dvar_e = omega
        dvar_e[var_e == 0] = 0
        delta_lam = step * dlam
        delta_var_e = step * dvar_e
        delta_phi = step * dphi
        lam = lam - delta_lam
        var_e = var_e - delta_var_e
        phi = phi - delta_phi
        if max(np.max(np.abs(delta_lam)), np.max(np.abs(delta_phi)), np.max(np.abs(delta_var_e))) < tol:
            return np.round(lam, rdd), np.round(phi, rdd), np.round(var_e, rdd)
    warnings.warn('no coverage')
    return np.round(lam, rdd), np.round(phi, rdd), np.round(var_e, rdd)


if __name__ == '__main__':
    strt_tm = time.time()

    lam = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    ])

    data = np.loadtxt('ex5.6.dat')
    lam, phi, var_e = cfa(data, lam)
    # 因子载荷
    print(lam)
    # 误差方差
    print(np.diag(var_e))
    # 潜变量协方差矩阵
    print(phi)


    print('Used time: {}s.'.format(round(time.time()-strt_tm, 6)))
