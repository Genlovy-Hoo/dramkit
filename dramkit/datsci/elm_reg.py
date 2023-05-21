# -*- coding: utf-8 -*-

import numpy as np
from inspect import isfunction
from dramkit.gentools import isnull


class ELMRegressor(object):
    '''
    | 极限学习机，回归任务

    | 记输入为层X，输出层为Y，隐藏层为H，样本量为Nsmp、X特征数为NcolX、
    | 隐藏层节点数为n_hide、Y特征数为NcolY，则ELM的过程为：
    |     H(Nsmp*n_hide) = X(Nsmp*NcolX) * w(NcolX*n_hide) + b((Nsmp*1)*n_hide)
    |     Y(Nsmp*NcolY) = H(Nsmp*n_hide) * beta(n_hide*NcolY)
    
    | ELM的训练过程：W和b随机生成，beta则利用公式求解析解(beta = H的MP广义逆 * Y)

    References
    ----------
    - https://blog.csdn.net/m0_37922734/article/details/80424133
    - https://blog.csdn.net/qq_32892383/article/details/90760481
    - https://blog.csdn.net/qq_40360172/article/details/105175946
    '''

    def __init__(self, n_hide=10, func_act='softplus',
                 w_low=-1, w_up=1, b_low=-1, b_up=1,
                 c=None, random_state=5262):
        '''
        Parameters
        ----------
        n_hide : int
            隐层节点数
        func_act: str, function
            激活函数，可选['softplus', 'sigmoid', 'tanh']或自定义
        w_low : float
            输入层—>隐层权重w取值下限
        w_up : float
            输入层—>隐层权重w取值上限
        b_low : float
            输入层—>隐层偏置项b取值下限
        b_up : float
            输入层—>隐层偏置项b取值上限
        c : float, None
            正则化参数
        random_state : None, int
            随机数种子
        '''

        self.n_hide = n_hide # 隐藏层节点数

        # 输入层—>隐层权重w和偏置项b取值上下限
        self.w_low = w_low
        self.w_up = w_up
        self.b_low = b_low
        self.b_up = b_up

        self.w = '未初始化参数(shape: NcolX*{})'.format(n_hide)
        self.b = '未初始化参数(shape: 1*{})'.format(n_hide)
        self.beta = '未初始化参数(shape: {}*NcolY)'.format(n_hide)

        # 正则化参数
        self.c = c

        # 激活函数
        if func_act == 'softplus':
            self.func_act = self.softplus
        elif func_act == 'sigmoid':
            self.func_act = self.sigmoid
        elif func_act == 'tanh':
            self.func_act = self.tanh
        else:
            if isfunction(func_act):
                self.func_act = func_act
            else:
                raise ValueError('不能识别的激活函数，请检查！')

        # 其他参数
        self.random_state = random_state

    @staticmethod
    def sigmoid(x):
        '''sigmoid激活函数'''
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        '''softplus激活函数 '''
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        '''tanh激活函数'''
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def fit(self, x_train, y_train):
        '''
        模型训练

        Parameters
        ----------
        x_train : pd.DataFrame, np.array
            训练集输入，每行一个样本
        y_train : pd.DataFrame, np.array
            训练集输出，每行一个样本
        '''

        x_train, y_train = np.array(x_train), np.array(y_train)
        Nsmp, NcolX = x_train.shape[0], x_train.shape[1] # 样本数和特征数

        # 将标签转化为二维
        self.Yreshape = False
        if len(y_train.shape) == 1:
            self.Yreshape = True
            y_train = y_train.reshape(-1, 1)

        # 随机数种子
        if isnull(self.random_state):
            rnd_w = np.random.RandomState()
            rnd_b = np.random.RandomState()
        else:
            rnd_w = np.random.RandomState(self.random_state)
            rnd_b = np.random.RandomState(self.random_state)

        # 输入层——>隐藏层权重w随机化
        self.w = rnd_w.uniform(self.w_low, self.w_up, (NcolX, self.n_hide))
        # 输入层——>隐藏层偏置b随机化
        self.b = rnd_b.uniform(self.b_low, self.b_up, (1, self.n_hide))
        Bhide= np.ones([Nsmp, self.n_hide]) * self.b

        # 隐层计算
        Hide = np.matrix(self.func_act(np.dot(x_train, self.w) + Bhide))

        # beta计算
        if isnull(self.c):
            iMP = np.linalg.pinv(Hide) # Moore–Penrose广义逆
            self.beta = np.dot(iMP, y_train)
        else:
            Hide_ = np.dot(Hide.T, Hide) + Nsmp / self.c
            iMP = np.linalg.pinv(Hide_) # Moore–Penrose广义逆
            iMP_ = np.dot(iMP, Hide.T)
            self.beta = np.dot(iMP_, y_train)

        return self

    def predict(self, x):
        '''模型预测，x每行为一个待预测样本'''

        Nsmp = x.shape[0]
        Bhide = np.ones([Nsmp, self.n_hide]) * self.b
        Hide = np.matrix(self.func_act(np.dot(x, self.w) + Bhide))
        y_pred = np.array(np.dot(Hide, self.beta))

        if self.Yreshape:
            y_pred = y_pred.reshape(-1)

        return y_pred

#%%
if __name__ == '__main__':
    import pandas as pd
    import sklearn.datasets as datasets
    from sklearn.model_selection import train_test_split as tts
    from dramkit import TimeRecoder
    from dramkit.datsci.stats import mape

    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['font.serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt


    tr = TimeRecoder()

    #%%
    # 产生数据集
    X = np.linspace(0, 20, 200)
    noise = np.random.normal(0, 0.08, 200)
    y = np.sin(X) + np.cos(0.5*X) + noise
    # 转化成二维形式
    X = np.array(X).reshape(-1, 1)

    func_act = 'softplus'
    w_low, w_up, b_low, b_up, = -1, 1, -1, 1
    c = None
    random_state = 5262

    Xtest = np.linspace(0, 20, 200).reshape(-1, 1)
    Ytest = np.sin(Xtest) + np.cos(0.5*Xtest)
    Ytest = Ytest.reshape(-1)
    plt.figure(figsize=(12, 7))
    plt.plot(Xtest, Ytest, 'or', label='ori') # 原始数据散点图
    # plt.plot(X, y, 'or', label='ori') # 原始数据散点图
    color = ['g', 'b', 'y', 'c', 'm'] # 不同隐藏层线条设置不同的颜色
    # 比较不同隐藏层拟合效果
    j=0
    for i in range(5, 30, 5):
        mdlELM = ELMRegressor(n_hide=i, func_act=func_act,
                         w_low=w_low, w_up=w_up, b_low=b_low, b_up=b_up,
                         c=c, random_state=random_state)
        mdlELM = mdlELM.fit(X, y)
        Ypre = mdlELM.predict(Xtest)
        plt.plot(Xtest, Ypre, color[j], label='n_hide_'+str(i))
        plt.title('ELM regression test')
        plt.xlabel('x')
        plt.ylabel('y')
        j+=1

        vMAPE = mape(Ytest, Ypre)
        print('n_hide - %d, mape：%f' % (i, vMAPE))
    plt.legend(loc=0)
    plt.show()

    #%%
    # data = datasets.load_boston()
    data = datasets.load_diabetes()
    X = pd.DataFrame(data['data'], columns=data.feature_names)
    # Y = pd.DataFrame(data['target'], columns=['y'])
    Y = pd.Series(data['target'])

    Xtrain, Xtest, Ytrain, Ytest = tts(X, Y, test_size=0.4,
                                        random_state=5262)

    func_act = 'softplus'
    w_low, w_up, b_low, b_up, = -1, 1, -1, 1
    c = None
    random_state = 5262

    for n_hide in range(1, 50, 2):
        mdlELM = ELMRegressor(n_hide=n_hide, func_act=func_act,
                          w_low=w_low, w_up=w_up, b_low=b_low, b_up=b_up,
                          c=c, random_state=random_state)

        mdlELM = mdlELM.fit(Xtrain, Ytrain)

        Ypre = mdlELM.predict(Xtest)

        vMAPE = mape(Ytest, Ypre)
        print('n_hide - %d, mape：%f' % (n_hide, vMAPE))

        # plt.figure(figsize=(12, 7))
        # plt.plot(Ytest.reset_index(drop=True), '.-b', label='Ytrue')
        # plt.plot(Ypre, '.-r', label='Ypre')
        # plt.legend(loc=0)
        # plt.show()

    #%%
    tr.used()
