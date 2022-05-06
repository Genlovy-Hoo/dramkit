#-*- coding:utf-8 -*-

if __name__ == '__main__':

    from scipy.integrate import odeint
    import numpy as np
    from matplotlib import pyplot as pl

    #解决matplotlib显示中文乱码问题
    pl.rcParams['font.sans-serif'] = ['SimHei']
    pl.rcParams['axes.unicode_minus'] = False


    def gini(x, w=None):
        # The rest of the code requires numpy arrays.
        x = np.asarray(x)
        if w is not None:
            w = np.asarray(w)
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_w = w[sorted_indices]
            # Force float dtype to avoid overflows
            cumw = np.cumsum(sorted_w, dtype=float)
            cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
            return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                    (cumxw[-1] * cumw[-1]))
        else:
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


    def gini_coef(series):
        '''
        计算序列series（pd.Series）的基尼系数
        注：貌似结果不能满足当财富全部一人所有时基尼系数为1
        https://www.zhihu.com/question/20219466
        '''
        cumsums = pd.Series([0]).append(series.sort_values()).cumsum()
        Csum = cumsums.iloc[-1]
        xarray = np.array(range(0, len(cumsums))) / (len(cumsums)-1)
        yarray = cumsums / Csum
        B = np.trapz(yarray, x=xarray) # 曲线下方面积
        A = 0.5 - B
        return A / (A + B)


    fig, ax = pl.subplots()

    #计算基尼系数
    def Gini():
        # 计算数组累计值,从 0 开始
        # wealths = [1.5, 2, 3.5, 10, 4.2, 2.1, 1.1, 2.2, 3.1, 5.1, 9.5, 9.7, 1.7, 2.3, 3.8, 1.7, 2.3, 5, 4.7, 2.3, 4.3, 12]
        wealths = [0.1,0.,0.,0]
        cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
        # 取最后一个，也就是原数组的和
        sum_wealths = cum_wealths[-1]
        # 人数的累积占比
        xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)

        # 均衡收入曲线
        upper = xarray
        # 收入累积占比
        yarray = cum_wealths / sum_wealths
        # 绘制基尼系数对应的洛伦兹曲线
        ax.plot(xarray, yarray)
        ax.plot(xarray, upper)
        ax.set_xlabel(u'人数累积占比')
        ax.set_ylabel(u'收入累积占比')
        pl.show()
        # 计算曲线下面积的通用方法
        B = np.trapz(yarray, x=xarray)
        # 总面积 0.5
        A = 0.5 - B
        G = A / (A + B)
        return G

    a=Gini()
    print(a)
