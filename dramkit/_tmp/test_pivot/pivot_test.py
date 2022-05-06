# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

if __name__ == '__main__':
    na = [np.nan]
    a = pd.DataFrame({'name': ['Saliy', 'Saliy', 'Jeff', 'Jeff',
                               'Roger', 'Roger', 'Karen', 'Karen',
                               'Brain', 'Brain'],
                      'test': ['midterm', 'final'] * 5,
                      'class1': ['A', 'C'] + na*6 + ['B', 'B'],
                      'class2': na*2 + ['D', 'E', 'C', 'A'] + na*4,
                      'class3': ['B', 'C'] + na*4 + ['C', 'C'] + na*2,
                      'class4': na*2 + ['A', 'C'] + na*2 + ['A', 'A'] + na*2,
                      'class5': na*4 + ['B', 'A'] + na*2 + ['A', 'C']})

    a1 = pd.DataFrame(a.set_index(['name', 'test']).stack())
    a1.columns = ['grade']
    a1.index.names = ['name', 'test', 'class']
    a1 = a1.reset_index()
