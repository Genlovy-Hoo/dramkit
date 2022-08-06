# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from dramkit import plot_series
    from sklearn.linear_model import LinearRegression as lr
    from dramkit.datsci.stats import cal_linear_reg_r
    
    
    Plant = lambda x: 5*x + 2    
    
    ss, N = [], 1000
    for a in tqdm(range(N)):
        xtrain = np.array(range(1, 100))
        ytrain = Plant(np.array(xtrain))
        np.random.seed(None)
        ytrain = ytrain + np.random.randn(ytrain.shape[0])
        
        # df = pd.DataFrame({'x': xtrain, 'y': ytrain})
        # df.set_index('x', inplace=True)
        # plot_series(df, {'y': ('.k')})
        
        
        k, b = cal_linear_reg_r(ytrain, xtrain)
        FFC = lambda x: k*x + b
        
        k_, b_ = cal_linear_reg_r(xtrain, ytrain)
        FBC = lambda y : k_*y + b_
        
        xtrain_pre = FBC(ytrain)
        xdif = xtrain - xtrain_pre
        mdl = lr().fit(np.concatenate(
                        (ytrain.reshape((-1, 1)),
                         xdif.reshape((-1, 1))),
                        axis=1),
                       xtrain)
        
        FEC = lambda y, e: mdl.predict(np.array([[y, e]]))
    
    
        x = 5
        y = Plant(x)
        
        ypre0 = FFC(x)
        ypre = FFC(x)
        x_ = FBC(ypre)
        for _ in range(100):
            # print(y, ypre)
            x_ = (x + x_) / 2
            ypre = FFC(x_)
            x_ = FBC(ypre) 
            e = x - x_
            x_ = FEC(ypre, e)[0]
        # print(y, ypre0, ypre)
        if abs(y-ypre0) > abs(y-ypre):
            # print(True)
            ss.append(1)
        else:
            # print(False)
            ss.append(0)
    print('\n\n', sum(ss) / N)
        
