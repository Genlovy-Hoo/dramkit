# -*- coding: utf-8 -*-

# https://pypi.org/project/backtrader/

if __name__ == '__main__':

    from datetime import datetime
    import backtrader as bt

    class SmaCross(bt.SignalStrategy):
        def __init__(self):
            sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
            crossover = bt.ind.CrossOver(sma1, sma2)
            self.signal_add(bt.SIGNAL_LONG, crossover)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    data0 = bt.feeds.YahooFinanceData(dataname='MSFT',
                                      fromdate=datetime(2020, 1, 1),
                                      todate=datetime(2021, 12, 31))
    cerebro.adddata(data0)

    cerebro.run()
    cerebro.plot()
