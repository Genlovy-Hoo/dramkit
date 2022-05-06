# -*- coding: utf-8 -*-

if __name__ == '__main__':
    Plant = lambda x: x + 2
    FFC = lambda x: 1.1 * x + 1.9
    FBC = lambda y: 0.9*y - 1.8

    x = 5
    x_ = x
    y = Plant(x)

    for k in range(100):
        ypre = FFC(x_)
        x_ = FFC(ypre)
        x_ = (x + x_) / 2
        print(y, ypre)
