# -*- coding: utf-8 -*-

'''
| 频繁项集挖掘数据集和算法研究资料参考：
| http://fimi.uantwerpen.be/
| http://fimi.uantwerpen.be/data/
| http://fimi.uantwerpen.be/src/
'''


import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

from dramkit.datsci.apriori import get_CkSup_and_FreqLk_all


def gen_rules(FreqSets, setSups, min_conf=0.7):
    '''
    | 生成关联规则并计算置信度和提升度
    | FreqSets(list)为所有频繁项集列表
    | setSups(dict)为项集支持度，FreqSets中的所有元素必须出现在setSups的keys中
    | (注：输入参数FreqSets和setSups格式同
      :func:`dramkit.datsci.apriori.get_CkSup_and_FreqLk_all` 函数的输出)
    '''

    # 构建可能的规则
    nakeRules = []
    subFreSets = []
    for FreqSet in FreqSets:
        for subFreSet in subFreSets:
            if subFreSet.issubset(FreqSet):
                rule = (FreqSet-subFreSet, subFreSet)
                if rule not in nakeRules:
                    nakeRules.append(rule)
        subFreSets.append(FreqSet)

    # 以最小置信度为条件筛选规则
    rules = []
    for preSet, postSet in nakeRules:
        Sup = setSups[preSet | postSet]
        preSup = setSups[preSet]
        Conf = Sup / preSup
        if Conf > min_conf:
            postSup = setSups[postSet]
            Lift = Conf / postSup
            rules.append((preSet, postSet, preSup, postSup, Sup, Conf, Lift))

    return rules


def apriori_dramkit(dataset, min_sup, min_conf):
    '''
    | Apriori主函数
    | dataset为list数据集，每个元素为一个样本(由不同项组成的事务，也为list)
    | min_sup, min_conf分别为最小支持度和最小置信度阈值
    | 返回list格式规则列表rules，每条规则包含：
    |     [前件, 后件, 前件支持度, 后件支持度, 支持度, 置信度, 提升度]
    '''

    FreqLk_all, CkSup_all = get_CkSup_and_FreqLk_all(dataset, min_sup)
    FreqSets = [y for x in FreqLk_all for y in x ]
    rules = gen_rules(FreqSets, CkSup_all, min_conf)

    return rules


def rules2df(rules, joiner='&'):
    '''
    将列表格式rules（gen_rules函数的输出格式）转存为pd.DataFrame格式，
    joiner设置项集元素之间的文本连接符
    '''

    df_rules = []
    for preSet, postSet, preSup, postSup, Sup, Conf, Lift in rules:
        preSet = (' ' + joiner + ' ').join([str(x) for x in list(preSet)])
        postSet = (' ' + joiner + ' ').join([str(x) for x in list(postSet)])
        df_rules.append([preSet, postSet, preSup, postSup, Sup, Conf, Lift])

    df_rules = pd.DataFrame(df_rules)
    df_rules.columns = ['前件', '后件', '前件支持度', '后件支持度', '支持度',
                        '置信度', '提升度']

    df_rules.sort_values(['支持度', '置信度', '前件', '后件'],
                         ascending=[False, False, True, True],
                         inplace=True)

    return df_rules


def arpiori_mlx(dataset, min_sup, min_conf, joiner='&'):
    '''
    | 调用mlxtend的Apriori算法实现关联规则挖掘
    | dataset为list数据集，每个元素为一个样本(由不同项组成的事务，也为list)
    | min_sup, min_conf分别为最小支持度和最小置信度阈值
    | 返回pd.DataFrame结果
    | joiner设置项集元素之间的文本连接符
    '''

    tranEncoder = TransactionEncoder()
    dataAry = tranEncoder.fit(dataset).transform(dataset)
    df = pd.DataFrame(dataAry, columns=tranEncoder.columns_)
    FreqSets = apriori(df, min_support=min_sup, use_colnames=True)
    rules = association_rules(FreqSets, min_threshold=min_conf)

    rules['antecedents'] = rules['antecedents'].apply(lambda x:
                        (' ' + joiner + ' ').join([str(y) for y in list(x)]))
    rules['consequents'] = rules['consequents'].apply(lambda x:
                        (' ' + joiner + ' ').join([str(y) for y in list(x)]))
    rules.sort_values(['support', 'confidence', 'antecedents', 'consequents'],
                      ascending=[False, False, True, True], inplace=True)

    return rules


def fpgrowth_mlx(dataset, min_sup, min_conf, joiner='&'):
    '''
    | 调用mlxtend的FP-growth算法实现关联规则挖掘
    | dataset为list数据集，每个元素为一个样本(由不同项组成的事务，也为list)
    | min_sup, min_conf分别为最小支持度和最小置信度阈值
    | 返回pd.DataFrame结果
    | joiner设置项集元素之间的文本连接符
    '''

    tranEncoder = TransactionEncoder()
    dataAry = tranEncoder.fit(dataset).transform(dataset)
    df = pd.DataFrame(dataAry, columns=tranEncoder.columns_)
    FreqSets = fpgrowth(df, min_support=min_sup, use_colnames=True)
    rules = association_rules(FreqSets, min_threshold=min_conf)

    rules['antecedents'] = rules['antecedents'].apply(lambda x:
                        (' ' + joiner + ' ').join([str(y) for y in list(x)]))
    rules['consequents'] = rules['consequents'].apply(lambda x:
                        (' ' + joiner + ' ').join([str(y) for y in list(x)]))
    rules.sort_values(['support', 'confidence', 'antecedents', 'consequents'],
                      ascending=[False, False, True, True], inplace=True)

    return rules


if __name__ == '__main__':
    # test1 ------------------------------------------------------------------
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    # dataset = [['菜品2', '菜品4', '菜品3'], ['菜品1', '菜品5'], ['菜品1', '菜品4'],
    #            ['菜品2', '菜品1', '菜品4', '菜品5'], ['菜品2', '菜品1'],
    #            ['菜品1', '菜品4'], ['菜品2', '菜品1'],
    #            ['菜品2', '菜品1', '菜品4', '菜品3'], ['菜品2', '菜品1', '菜品4'],
    #            ['菜品2', '菜品4', '菜品3']]

    min_sup = 0.3
    min_conf = 0.5

    # Apriori
    RulesApriHoo = apriori_dramkit(dataset, min_sup, min_conf)
    RulesApriHoodf = rules2df(RulesApriHoo)

    # 使用mlxtend的算法
    RulesApri_mlx = arpiori_mlx(dataset, min_sup, min_conf)
    RulesFpGrow_mlx = fpgrowth_mlx(dataset, min_sup, min_conf)


    # test2 ------------------------------------------------------------------
    # from dramkit import TimeRecoder
    # from dramkit import load_text

    # fpath = '../../../DataScience/dataSets/Frequent_Itemset_Mining_Dataset/T10I4D100K.dat'
    # dataset = load_text(fpath, sep=' ', keep_header=False, to_pd=False)

    # min_sup = 0.01
    # min_conf = 0.01

    # # Apriori
    # tr = TimeRecoder()
    # RulesApriHoo = apriori_dramkit(dataset, min_sup, min_conf)
    # RulesApriHoodf = rules2df(RulesApriHoo)
    # tr.used()

    # # 使用mlxtend的算法
    # tr = TimeRecoder()
    # RulesApri_mlx = arpiori_mlx(dataset, min_sup, min_conf)
    # tr.used()

    # tr = TimeRecoder()
    # RulesFpGrow_mlx = fpgrowth_mlx(dataset, min_sup, min_conf)
    # tr.used()
