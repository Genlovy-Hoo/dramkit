# -*- coding: utf-8 -*-

# https://blog.csdn.net/weixin_43933169/article/details/104343441

import matplotlib.pyplot as plt

if __name__ == '__main__':
    class Pid1():
        '''位置式'''
        def __init__(self, exp_val, kp, ki, kd):
            self.KP = kp
            self.KI = ki
            self.KD = kd
            self.exp_val = exp_val # 期望值
            self.now_val = 0 # 当前值
            self.sum_err = 0 # 累计误差（积分）？
            self.now_err = 0 # 误差
            self.last_err = 0

        def cmd_pid(self):
            self.last_err = self.now_err
            self.now_err = self.exp_val - self.now_val
            self.sum_err += self.now_err
            self.now_val = self.KP * (self.exp_val - self.now_val) + \
                           self.KI * self.sum_err + \
                           self.KD * (self.now_err - self.last_err)
            return self.now_val


    exp_val = 1000 # 目标值
    kp, ki, kd =  0.1, 0.15, 0.1
    pid_val = []
    my_Pid = Pid1(exp_val, kp, ki, kd)
    L = 100
    for i in range(0, L):
        pid_val.append(my_Pid.cmd_pid())
    plt.plot(pid_val)
    plt.show()


    import matplotlib.pyplot as plt


    class Pid2():
        '''增量式'''
        def __init__(self, exp_val, kp, ki, kd):
            self.KP = kp
            self.KI = ki
            self.KD = kd
            self.exp_val = exp_val
            self.now_val = 0
            self.now_err = 0
            self.last_err = 0
            self.last_last_err = 0
            self.change_val = 0

        def cmd_pid(self):
            self.last_last_err = self.last_err
            self.last_err = self.now_err
            self.now_err = self.exp_val - self.now_val
            self.change_val = self.KP * (self.now_err - self.last_err) + \
                              self.KI * self.now_err + \
                              self.KD * (self.now_err - 2 * self.last_err + \
                                         self.last_last_err)
            self.now_val += self.change_val
            return self.now_val


    pid_val = []
    my_Pid = Pid2(1000000, 0.1, 0.15, 0.1)
    for i in range(0, 30):
        pid_val.append(my_Pid.cmd_pid())
    plt.plot(pid_val)
    plt.show()
