# -*- coding: utf-8 -*-

# https://zhuanlan.zhihu.com/p/141871796

if __name__ == '__main__':

    import numpy as np


    # def MPCcontroller(pos_ref, pos, vel):
    #     # 参数设置
    #     m = 1.05 # m, 增加了5% 作为建模误差
    #     T = 0.01 # 控制周期10ms
    #     p = 45 # 控制时域（预测时域）
    #     Q = 10*eye(2*p) # 状态误差权重
    #     W = 0.0001*eye(p) # 控制输出权重
    #     umax = 100 # 控制量限制，即最大的力
    #     Rk = np.zeros((2*p,1)) # 参考值序列
    #     Rk[1:2:end] = pos_ref # 参考位置由函数参数指定
    #     Rk[2:2:end] = vel # 参考速度跟随实际速度
    #     # 构建中间变量
    #     xk = [pos;vel];    # xk
    #     A_ = [1 T;0 1];    # 离散化预测模型参数A
    #     B_ = [0;T/m];      # 离散化预测模型参数B
    #     psi   = zeros(2*p,2); # psi
    #     for i=1:1:p
    #         psi(i*2-1:i*2,1:2)=A_^i;
    #     end
    #     theta = zeros(2*p,p);     # theta
    #     for i=1:1:p
    #         for j=1:1:i
    #             theta(i*2-1:i*2,j)=A_^(i-j)*B_;
    #         end
    #     end
    #     E = psi*xk-Rk;            # E
    #     H = 2*(theta'*Q*theta+W); # H
    #     f = (2*E'*Q*theta)';      # f
    #     # 优化求解
    #     coder.extrinsic('quadprog');
    #     Uk=quadprog(H,f,[],[],[],[],-umax,umax);
    #     # 返回控制量序列第一个值
    #     u = 0.0;                # 指定u的类型
    #     u = Uk(1);              # 提取控制序列第一项

    #     return u
