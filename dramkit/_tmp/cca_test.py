# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import  CCA

    #设置随机种子
    np.random.seed(0)
    n = 500
    l1 = np.random.normal(size=n)
    l2 = np.random.normal(size=n)
    # print(l1.shape, l2.shape)
    latents = np.array([l1, l1, l2, l2]).T
    latents1 = np.array([l1, l1, l2, l2, l1, l1, l2, l2]).T
    # print(latents.shape)
    #加噪处理
    X = latents1 + np.random.normal(size=8 * n).reshape((n, 8))
    Y = latents + np.random.normal(size=1 * n).reshape((n, 1))
    print(X.shape)
    #划分数据集
    X_train = X[:n // 2]
    Y_train = Y[:n // 2]
    X_test = X[n // 2:]
    Y_test = Y[n // 2:]
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(X.T.shape)
    # 打印相关矩阵
    #保留小数点后2位
    print("Corr(X)")
    print(np.round(np.corrcoef(X.T), 2))
    print("Corr(Y)")
    print(np.round(np.corrcoef(Y.T), 2))
    #建立模型
    cca = CCA(n_components=1)
    #训练数据
    cca.fit(X_train, Y_train)
    #降维操作
    X_train_r, Y_train_r = cca.transform(X_train, Y_train)
    # print(X_train_r.shape, Y_train_r.shape)
    # print(X_train_r[:, 1].shape)
    X_test_r, Y_test_r = cca.transform(X_test, Y_test)
    print('test corr = %.2f' % np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
    # print(X_test_r.shape, Y_test_r.shape)
    # print(X_test_r[:, 1].shape)
    #画散点图
    plt.figure('CCA', facecolor='lightgray')
    plt.title('CCA', fontsize=16)
    plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train_data",
                marker="o", c="dodgerblue", s=25, alpha=0.8)
    plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test_data",
                marker="o", c="orangered", s=25, alpha=0.8)
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.tight_layout()
    plt.show()
