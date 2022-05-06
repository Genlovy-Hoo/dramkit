# -*- coding: utf-8 -*-

# https://pypi.org/project/pyfinance/

if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    from pyfinance import TSeries

    np.random.seed(444)

    # Normally distributed with 0.08% daily drift term.
    s = np.random.rand(400)  + 0.0008
    idx = pd.date_range(start='2016', periods=len(s))  # default daily freq.
    ts = TSeries(s, index=idx)

    print(ts.head())
    print(ts.max_drawdown())
    print(ts.sharpe_ratio())
