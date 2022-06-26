# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from dramkit.datsci.entropy_weight import entropy_weight


def topsis(df, weight=None):
    '''
    TOPSIS评价方法

    Parameters
    ----------
    df : pd.DataFrame
        每行一个样本，每列一个指标，为每个样本计算综合得分
        
        Note
        ----
        df应保证所有指标都是正向的，且已经预处理了无效值
    weight : list
        顺序与df.columns对应的指标权重列表，若为None，则默认采用熵值法计算权重

    Returns
    -------
    score : pd.DataFrame
        每个样本综合得分指数，列名为'score'
    mid_info : tuple
        返回中间过程数据(result, Z, weight):

        | - result: df规范化之后的数据，加上'正dis', '负dis', 'score', 'rank'四列，
          '正dis'和'负dis'分别为样本距正理想解和负理想解的距离
        | - Z: 正理想解和负理想解
        | - weight: 指标权重列表

    References
    -----------
    https://zhuanlan.zhihu.com/p/37738503
    '''

    # 权重
    if weight is None:
        weight, _ = entropy_weight(df, neg_cols=[], score_type=None)
        weight = list(weight['weight'])
    weight = np.array(weight)

    # 规范化
    df = df / np.sqrt((df ** 2).sum())

    # 最优最劣方案（正理想解和负理想解）
    Z = pd.DataFrame([df.max(), df.min()], index=['正理想解', '负理想解'])

    # 与最优最劣方案距离
    result = df.copy()
    result['正dis'] = np.sqrt(((df-Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
    result['负dis'] = np.sqrt(((df-Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

    # 样本综合得分指数
    result['score'] = result['负dis'] / (result['负dis'] + result['正dis'])
    result['rank'] = result.rank(ascending=False)['score'] # 得分越高rank越小
    score = result.reindex(columns=['score'])
    
    mid_info = (result, Z, weight)

    return score, mid_info


if __name__ == '__main__':
    from dramkit.datsci.preprocess import norm_range

    # df = pd.read_csv('../_test/EntValWeight_test.csv').dropna(how='any')
    # weight = None

    # df = pd.read_csv('../_test/GDP2015.csv', encoding='gbk').set_index('地区')
    # indexs = ['GDP总量增速', '人口总量', '人均GDP增速', '地方财政收入总额',
    #           '固定资产投资', '社会消费品零售总额增速', '进出口总额',
    #           '城镇居民人均可支配收入', '农村居民人均可支配收入']
    # df = df.reindex(columns=indexs).dropna(how='any')
    # weight = None

    df = pd.DataFrame(
            {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2],
              '生师比': [5, 6, 7, 10, 2],
              '科研经费': [5000, 6000, 7000, 10000, 400],
              '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]},
              index=['院校' + i for i in list('ABCDE')])
    # 师生比数据为区间型指标（落在某一个确定的区间最好）
    df['生师比'] = df['生师比'].apply(lambda x: norm_range(x, 5, 6, 2, 12))
    df['逾期毕业率'] = 1 / df['逾期毕业率'] # 逾期毕业率为极小型指标（越小越好）
    weight = [0.2, 0.3, 0.4, 0.1]
    # weight = None


    score, (result, Z, weight) = topsis(df, weight=weight)
    print(result[['score', 'rank']])
