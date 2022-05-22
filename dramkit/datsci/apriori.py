# -*- coding: utf-8 -*-

'''
| 关联规则-Apriori算法实现
|
| 关联规则分析（挖掘）用于发现项目或项目集合之间可能“频繁”出现的“紧密”关联模式（或因果结构），
| 如“由于某些事件的发生而引起另外一些事件的发生”之类的规则。“频繁”和“紧密”程度通过
  “支持度”和“置信度”来度量。
| 一个被津津乐道的案例是通过关联规则挖掘发现“买尿布的年轻父亲通常也会顺便为自己买啤酒”。
|
| 关联规则数据集示例：
| 编号, 交易清单
| 001, 可乐 & 鸡蛋 & 香肠
| 002, 可乐 & 尿布 & 啤酒
| 003, 可乐 & 尿布 & 啤酒 & 香肠
| 004, 尿布 & 啤酒
|
| 相关概念：
| 1. 事务：每一个样本（一条交易记录）称为一个事务，示例中的数据集包含四个事务。
| 2. 项：事务（交易记录）中的每一个组成部分（交易物品）称为一个项，例如可乐、鸡蛋等。
| 3. 项集：包含零个或多个项的集合叫做项集，例如{可乐, 鸡蛋, 香肠}。
| 4. k项集：包含k个项的项集叫做k-项集，{可乐}是1项集，{可乐, 鸡蛋}是2项集。
| 5. 支持度计数：一个项集在数据集不同事务中出现的次数。
     示例中{尿布, 啤酒}出现在事务002、003和004中，其支持度计数为3。
| 6. 支持度：支持度计数除以总的事务数。
     示例中{尿布, 啤酒}的支持度为3÷4=0.75（有75%的人同时买了尿布和啤酒）。
| 7. 频繁项集：支持度不小于给定阈值的项集就叫做频繁项集。
     若阈值设为0.7，示例中{尿布, 啤酒}的支持度是0.75 > 0.7，是频繁项集。
| 8. 前件和后件：对于规则{尿布}→{啤酒}，{尿布}叫做前件，{啤酒}叫做后件。
| 9. 置信度：对于规则{X}→{Y}，其置信度为Conf=P(Y|X)=P(XY)/P(X)（条件概率公式）。
     示例中规则{尿布}→{啤酒}的支持度为3÷3=1（即{尿布, 啤酒}的支持度计数除以
     {尿布}的支持度计数），置信度为100%说明买了尿布的人100%也买了啤酒）。
| 10. 强关联规则：大于或等于最小支持度阈值和最小置信度阈值的规则叫做强关联规则。
| 11. 提升度：对于规则{X}→{Y}，其提升度为Lift=P(Y|X)/P(Y)=Conf/P(Y)。
      提升度度量了后件分别在以前件为条件和不以前件为条件时发生概率的比值。
      提升度大于1表示关联规则有效（在X发生的情况下Y发生的概率比Y本身发生的概率大），
      提升度小于等于1表示，关联规则无效（在X发生的情况下Y发生的概率不变或者反而变小了）。
|
| 关联规则挖掘的最终目的就是要找出有效的强关联规则。
|
| 关联规则原理其实很简单，就是贝叶斯公式及其拓展。
| 关联规则挖掘难点在于如何高效快速地从大量数据中发现有效的强关联规则，
  因为事务中不同项的组合可以形成的规则数量很庞大。
| (假设有n个商品，潜在的关联规则数量为n*(n-1)+n*(n-1)*(n-2)+···，枚举法几乎不能用)
|
| Apriori算法基于两个定律，能有效提升挖掘效率。
|
| **Apriori定律1** ：如果一个集合是频繁项集，则它的所有子集都是频繁项集。
|     eg. 若集合{A, B}是频繁项集，由于出现{A, B}的地方一定会出现{A}和{B}，
          因此{A}和{B}出现的次数一定不小于{A, B}，故{A}和{B}一定是频繁项集。
|
| **Apriori定律2** ：如果一个集合不是频繁项集，则它的所有超集都不是频繁项集。
|     eg. 若集合{A}不是频繁项集，由于出现{A}的地方不一定出现{A, B}，
          因此{A, B}出现的次数一定不大于{A}，故{A}的超集{A, B}一定不是频繁项集。
|
| 基于两个定律，Apriori思路是以最小支持度为条件先找出所有频繁项集，然后由频繁项集构建规则，
  再计算置信度和支持度筛选出有效强关联规则。寻找频繁项集算法流程如下：
|
| step1. 扫描整个数据集，得到所有出现过的项，作为候选频繁1项集C(1)。
| step2. 对k >= 1，筛选出频繁k项集L(k)。
|     step2.1. 计算候选频繁k项集的支持度（需要扫描整个数据集）。
|     step2.2. 去除候选频繁k项集中支持度低于阈值的项集（依据定律2），
               得到频繁k项集L(k)并保存，如果得到的频繁k项集L(k)为空集或只有一项，
               则返回所有频繁项集，算法结束。
|     step2.3. 基于频繁k项集，项集之间取并集生成候选频繁k+1项集C(k+1)。
| step3. 令k=k+1，转入step2。
| 
| 得到所有频繁项集之后，再在频繁项集的子集之间构建规则并计算置信度和提升度，
  筛选出有效强关联规则即可。
|
| 缺点:
| Aprior算法每轮迭代筛选频繁k项集都要扫描整个数据集，因此在数据集很大，
  数据种类很多的时候，算法效率仍然很低。
|
| 参考:
| https://blog.csdn.net/qq_23860475/article/details/80824568
| https://www.cnblogs.com/pinard/p/6293298.html
| https://www.jianshu.com/p/26d61b83492e
| https://blog.csdn.net/zllnau66/article/details/81534368
| https://www.cnblogs.com/MaggieForest/p/12176915.html
| https://www.jianshu.com/p/fba9e41334a8
| https://blog.csdn.net/tangyudi/article/details/88822705
'''

def gen_C1(dataset):
    '''
    | 生成初始项集C1，每个项集以python不变集合frozenset格式保存。
    | 数据集dataset为list格式，每个list元素为一个样本（即由不同项组成的事务）。
    '''

    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    C1 = list(map(frozenset, C1))

    return C1


def get_CkSup_and_FreqLk(dataset, Ck, min_sup):
    '''
    | 筛选频繁项集
    | 扫描数据集dataset(dataset为list格式，每条事务记录以set或frozenset格式保存)，
    | 计算Ck中项集的支持度CkSup，并筛选出Ck中不小于最小支持度min_sup的频繁项集列表FreqLk。
    '''

    CkSupCnt = {} # Ck中各项集的支持度计数
    for transaction in dataset:
        for ItemSet in Ck:
            if ItemSet.issubset(transaction):
                if not ItemSet in CkSupCnt:
                    CkSupCnt[ItemSet] = 1
                else:
                    CkSupCnt[ItemSet] += 1

    numItems = len(dataset)
    CkSup = {} # Ck中各项集的支持度
    FreqLk = [] # Ck中支持度大于等于最小支持度的项集
    for ItemSet in CkSupCnt:
        support = CkSupCnt[ItemSet] / numItems
        CkSup[ItemSet] = support
        if support >= min_sup:
            FreqLk.insert(0, ItemSet)

    return CkSup, FreqLk


def gen_Ck_from_FreqLk_1(FreqLk_1, k):
    '''
    由频繁k-1项集FreqLk_1生成k项集Ck
    '''

    Ck = []
    nLk_1 = len(FreqLk_1)

    for i in range(nLk_1):
        for j in range(i+1, nLk_1):
            # 这里排序并比较不同项集的前部分，只有前部分相同时才合并成一个新项集
            # 这是为了避免生成不必要的不满足最小支持度的项集
            # 比如{1, 2}和{2, 3}在频繁2项集中，但是{1, 3}不在频繁2项集中
            # 根据Apriori定律2，{1, 3}的任何超集都不是频繁项集，
            # 因此{1, 2}和{2, 3}的并集{1, 2, 3}肯定不是频繁项集
            # 当只有两个项集的前部分相同时才合并成新项集可以避免{1, 2, 3}出现
            pre1 = list(FreqLk_1[i])[:k-2]
            pre2 = list(FreqLk_1[j])[:k-2]
            pre1.sort()
            pre2.sort()

            if pre1 == pre2:
                Ck.append(FreqLk_1[i] | FreqLk_1[j])

    return Ck


def get_CkSup_and_FreqLk_all(dataset, min_sup=0.5):
    '''
    Apriori算法寻找最大频繁K项集

    Parameters
    ----------
    dataset : list
        数据集，list中每个元素为一个样本（即由不同项组成的事务，也为list）
    min_sup : float
        最小支持度阈值

    Returns
    -------
    FreqLk_all : list
        所有频繁项集FreqLk，每个元素均为频繁项集列表，
        如FreqLk_all[1]为所有频繁2项集列表。
    CkSup_all : dict
        所有Ck的支持度

        Note
        ----
        CkSup_all中保存了每一步Ck中所有项集的支持度，
        因此CkSup_all中并不是所有Ck全部都满足最小支持度的。
    '''

    C1 = gen_C1(dataset) # 初始项集C1
    dataset = list(map(frozenset, dataset))
    CkSup_all, FreL1 = get_CkSup_and_FreqLk(dataset, C1, min_sup) # FreL1为频繁1项集

    FreqLk_all = [FreL1]
    k = 2
    while len(FreqLk_all[k-2]) > 0:
        Ck = gen_Ck_from_FreqLk_1(FreqLk_all[k-2], k) # 生成k项集Ck
        # 从Ck中筛选频繁k项集FreqLk
        CkSup, FreqLk = get_CkSup_and_FreqLk(dataset, Ck, min_sup)

        CkSup_all.update(CkSup) # 保存每一步Ck支持度
        FreqLk_all.append(FreqLk) # 保存所有频繁k项集

        k += 1

    return FreqLk_all, CkSup_all


if __name__ == '__main__':
    from pprint import pprint
    
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    # dataset = [['菜品2', '菜品4', '菜品3'], ['菜品1', '菜品5'], ['菜品1', '菜品4'],
    #            ['菜品2', '菜品1', '菜品4', '菜品5'], ['菜品2', '菜品1'],
    #            ['菜品1', '菜品4'], ['菜品2', '菜品1'],
    #            ['菜品2', '菜品1', '菜品4', '菜品3'], ['菜品2', '菜品1', '菜品4'],
    #            ['菜品2', '菜品4', '菜品3']]
    
    min_sup = 0.6
    
    FreqLk_all, CkSup_all = get_CkSup_and_FreqLk_all(dataset, min_sup)
    print('CkSup_all:')
    pprint(FreqLk_all)
    print('CkSup_all:')
    pprint(CkSup_all)
