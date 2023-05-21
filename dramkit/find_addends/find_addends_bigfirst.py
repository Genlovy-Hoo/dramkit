# -*- coding: utf-8 -*-

import time
# from .find_addends_utils import get_alts_sml, tol2side_x_eq_y, backfind_sml1st_index
# # from find_addends_utils import get_alts_sml, tol2side_x_eq_y, backfind_sml1st_index
from dramkit.find_addends.find_addends_utils import get_alts_sml
from dramkit.find_addends.find_addends_utils import tol2side_x_eq_y
from dramkit.find_addends.find_addends_utils import backfind_sml1st_index
from dramkit.logtools.utils_logger import logger_show


def find_addends_bigfirst(tgt_sum, alts, n_adds=None, check_alts=False,
                          add_num=None, tol_below=0.0, tol_above=0.0,
                          max_loop=1000000, global_loop=False, log_info=False,
                          save_process=False, logger=None):
    '''
    | 从给定的列表alts（可以有负数）中选取若干个加数之和最接近tgt_sum，大的备选数优先

    | 思路: 从大到小依次加入备选数
    |      若加入新值之后找不到理想解，则删除最后加入的值，继续添加下一个更小的备选数
    |      下一个备选数确定方式：
    |          当alts中只有正数时，剩下的数中与剩余和（目标和减去当前和）最接近的数
    |          当alts中有负数时，直接取比最后加进去的数更小的数（搜索速度会变慢很多）

    Parameters
    ----------
    tgt_sum : float, int
        目标和
    alts : list
        备选数列表
    n_adds : int
        限制备选加数个数为n_adds，默认无限制
    check_alts : bool
        | 是否检查alts（提前删除大于目标和的备选数），默认否
        | 注：该参数只有当alts没有负数时才起作用
    add_num : int
        在加起来和大于等于tgt_sum的基础上增加的备选数个数，意义同 
        :func:`dramkit.find_addends.find_addends_utils.get_alts_sml` 函数中的add_num参数，在check_alts起作用时才起作用
    tol_below : float
        下界误差，若目标和减去当前和小于等于tol_below，则结束搜索，默认为0.0
    tol_above : float
        上界误差，若当前和减去目标和小于等于bol_above，则结束搜索，默认为0.0
    max_loop : int
        最大搜索次数限制，默认一百万
    global_loop : bool
        是否将寻找下一个最优备选数时的循环次数计入总搜索次数
    log_info : bool
        是否打印搜索次数和时间等信息
    save_process : bool
        是否保存搜索过程
    logger : Logger
        logging日志记录器

    Returns
    -------
    choseds_best : list
        最优备选数列表
    choseds_addends : list
        最终备选数列表
    adds_process : list
        搜索中间过程
    '''

    adds_process = [] if save_process else None

    if log_info:
        start_time = time.monotonic()

    all_postive = all([x >= 0 for x in alts])

    if len(alts) == 0:
        return [], [], adds_process

    # 当目标和小于alts的最小值且alts没有负数时可直接返回最接近值
    if tgt_sum < min(alts) and all_postive:
        return [min(alts)], [], adds_process

    # 当alts全为正数时可提前删除大于目标和的备选数（有负数时不能）
    if check_alts and all_postive:
        alts = get_alts_sml(tgt_sum, alts, typeSort='descend', tol=tol_above,
                            add_num=add_num)

    alts.sort() # 升序（大的数优先进入）

    # 当alts全为正数时若目标和大于alts所有数之和，可直接返回（有负数时不能）
    if tgt_sum >= sum(alts) and all_postive:
        adds_process = alts if save_process else adds_process
        return alts, alts, adds_process

    # 初始化，idx_last记录最新进入的数的索引号
    idx_last = len(alts) - 1
    chosed_idxs = [idx_last]
    chosed_addends = [alts[idx_last]]
    if save_process:
        adds_process.append(chosed_addends.copy())
    choseds_best = []
    now_sum = alts[idx_last]

    # 搜索过程
    loop_count = 0
    while loop_count < max_loop:
        loop_count += 1

        # 更新最优解(此处取等号是为了避免更新的解满足n_adds限制但是最优解没有更新的情况)
        if abs(tgt_sum - now_sum) <= abs(tgt_sum - sum(choseds_best)):
            choseds_best = chosed_addends.copy()

        # 结束条件
        if n_adds is None:
            if tol2side_x_eq_y(now_sum, tgt_sum,
                               tol_below=tol_below, tol_above=tol_above):
                if log_info:
                    logger_show('找到最优解，结束搜索。', logger, 'info')
                break
        else:
            if tol2side_x_eq_y(now_sum, tgt_sum, tol_below=tol_below,
                         tol_above=tol_above) and len(choseds_best) == n_adds:
                if log_info:
                    logger_show('找到最优解，结束搜索。', logger, 'info')
                break

        # 无最优解（搜索到最小值且只剩它一个备选数），结束搜索
        if idx_last == 0 and len(chosed_idxs) == 1:
            if log_info:
                logger_show('无最优解，结束搜索。', logger, 'info')
            break

        # 刚好搜索到最小值且不是最优解，此时去掉最小值并更改最小值前面一个进去的值
        if idx_last == 0:
            idx_last = chosed_idxs[-2] - 1
            del chosed_idxs[-2:]
            chosed_idxs.append(idx_last)
            del chosed_addends[-2:]
            chosed_addends.append(alts[idx_last])
            if save_process:
                adds_process.append(chosed_addends.copy())
            now_sum = sum(chosed_addends)
            continue

        # 下一个备选数
        if not all_postive:
            idx_last -= 1
        else:
            if global_loop:
                idx_last, loop_count = backfind_sml1st_index(
                        tgt_sum-now_sum, alts[0:idx_last], tol=tol_above,
                        loop_count=loop_count)
            else:
                idx_last, _ = backfind_sml1st_index(tgt_sum-now_sum,
                                                    alts[0:idx_last],
                                                    tol=tol_above)

        # 保留最后一个加进去的数情况下找不到最优解，更改最后进去的那个数
        if idx_last < 0:
            idx_last = chosed_idxs[-1] - 1
            chosed_idxs[-1] = idx_last
            chosed_addends[-1] = alts[idx_last]
            if save_process:
                adds_process.append(chosed_addends.copy())
            now_sum = sum(chosed_addends)
            continue

        chosed_idxs.append(idx_last)
        chosed_addends.append(alts[idx_last])
        if save_process:
            adds_process.append(chosed_addends.copy())
        now_sum += alts[idx_last]

    if log_info:
        logger_show('loop_count: {}'.format(loop_count), logger, 'info')
        logger_show('used time {}s.'.format(round(time.monotonic()-start_time, 6)),
                    logger, 'info')

    return choseds_best, chosed_addends, adds_process


if __name__ == '__main__':
    alts = [17541875.0, 16901617.0, 10696518.0, 6388100.0, 2305050.0, 987414.0,
            783360.0, 632604.0, 628650.0, 609600.0, 601650.0, 600668.0] + \
           [590550.0, 599557.0, 628000.0, 628650.0, 628650.0, 628650.0,
            631825.0, 632000.0, 679450.0, 704850.0, 725479.0, 746125.0,
            774700.0, 784225.0, 796925.0, 809580.0, 930275.0, 930275.0,
            938770.0, 947937.0, 987425.0, 987425.0, 1015625.0, 1054100.0,
            1057275.0, 1057275.0, 1069975.0, 1089025.0, 1108525.0, 1111250.0,
            1133475.0, 1231900.0, 1250950.0, 1266825.0, 1270000.0, 1272275.0,
            1463675.0, 1485900.0, 1500000.0, 1500000.0, 1552575.0, 1654175.0,
            1993900.0, 2136592.0, 2222500.0, 2232025.0, 2471826.0, 2857500.0,
            6350000.0, 9503422.2, 9918683.4]
    tgt_sum = 58677106.0 # alts中第一个列表之和

    alts = [7440000.0, 7430992.0, 4190208.0, 3720000.0, 3720000.0, 2975256.0,
            2514720.0, 2496120.0, 2231475.0, 2228997.0, 2219001.0, 2213181.0,
            1749144.0, 1546776.0, 995669.0, 472550.0, 444160.0, 427452.0,
            394837.0] + \
           [401496.0, 412498.0, 418322.0, 420748.0, 427800.0, 429120.0,
            433200.0, 436728.0, 437628.0, 438368.0, 438370.0, 438861.0,
            440208.0, 440800.0, 441000.0, 444600.0, 446400.0, 450120.0,
            457475.0, 458602.0, 460956.0, 461107.0, 465152.0, 466604.0,
            471192.0, 472221.0, 472505.0, 483600.0, 486506.0, 492284.0,
            499366.0, 499813.0, 501160.0, 501952.0, 501979.0, 502425.0,
            503688.0, 509760.0, 519406.76, 523452.0, 525198.0, 526612.0,
            528504.0, 531405.0, 549442.0, 555768.0, 578530.93, 584000.0,
            584233.0, 584746.0, 587636.0, 588800.0, 591889.0, 595200.0,
            595944.0, 596373.0, 603460.0, 604964.59, 607560.0, 615930.0,
            621810.0, 642664.0, 645330.0, 654914.8, 678729.0, 680062.06,
            705535.0, 705859.0, 708994.0, 714850.0, 715360.4, 727736.0,
            731314.0, 738000.0, 738000.0, 739000.0, 739000.0, 739000.0,
            739794.0, 740000.0, 740000.0, 740000.0, 740000.0, 740000.0,
            741000.0, 746715.0, 750729.0, 766033.0, 767810.0, 784920.0,
            801021.0, 801150.0, 826584.0, 853024.0, 883048.0, 892254.0,
            932010.0, 1093500.0, 1106328.0, 1115730.0, 1116000.0, 1148988.0,
            1187942.0, 1459393.0, 1517016.0, 1522224.0]
    tgt_sum = 49410538.0 # alts中第一个列表之和

    alts = [1201008.0, 1254715.0, 1269351.0, 1277352.0, 1291173.81, 1317600.0,
            1317876.0, 1330352.0, 1353600.0, 1354412.0, 1370457.0, 1374766.0,
            1408844.75, 1439220.0, 1510600.0, 1524456.0, 1617486.0, 1672787.0,
            1681834.0, 1686519.0, 1687405.0, 1690206.0, 1695000.0, 1724995.0,
            1751435.0, 1785641.0, 1843309.36, 1856886.0, 1894970.0, 1898708.0,
            1912052.0, 1921500.0, 1929600.0, 1963981.2, 1969590.0, 2023032.0,
            2026415.0, 2193621.0, 2196599.0, 2199462.0, 2199786.0, 2209606.0,
            2209663.0, 2235424.0, 2240000.0, 2243847.0, 2249822.0, 2258363.0,
            2270664.28, 2322470.17, 2324000.0, 2340800.0, 2416128.0, 2574972.0,
            2727184.0, 2752425.0, 2759907.72, 2852438.0, 2866005.1, 2868857.0,
            2975417.2, 3292897.0, 3296537.0, 3298769.36, 3329329.0, 3433606.0,
            3444547.1, 449625.1, 4036216.0, 4415600.0, 4431336.72, 4637542.0,
            4641028.0, 5442861.0, 5499746.0, 5502825.0, 5505000.0, 5574654.0,
            5636556.0, 5774073.0, 5788190.0, 5810000.0, 9069410.0, 11291809.0,
            11504962.0, 11590342.0, 11618838.0, 11620000.0, 11620000.0]
    # alts = alts[-65:]
    tgt_sum = 71711814.0

    alts = [22, 15, 14, 13, 7, 6.1, 5, 21.5, 100]
    tgt_sum = 22 + 21 + 4.1

    alts = [200, 107, 100, 99, 98, 6, 5, 3, -1, -20, -25]
    tgt_sum = 100 + 6 - 25

    # alts = [100, -100, -105, -102, -25, -30, -1]
    # tgt_sum = -26

    alts = [10, 9, 8, 7, 6, 5]
    tgt_sum = 17

    # alts = [10, 9, 8, -7, -6, -5]
    # tgt_sum = 3

    # alts = [10, 7, 6, 3]
    # tgt_sum = 18

    alts = [10, 7, 6, 3]
    tgt_sum = 12


    n_adds = None
    check_alts = False
    add_num = None
    tol_below = 0.0
    tol_above = 0.0
    max_loop = 10000000
    global_loop = False
    log_info = True
    save_process = True
    logger = None


    best, choseds, mid = find_addends_bigfirst(tgt_sum, alts, n_adds=n_adds,
                                               check_alts=check_alts,
                                               add_num=add_num,
                                               tol_below=tol_below,
                                               tol_above=tol_above,
                                               max_loop=max_loop,
                                               global_loop=global_loop,
                                               log_info=log_info,
                                               save_process=save_process,
                                               logger=logger)

    print(' 最终结果:', choseds)
    print(' 最优结果:', best, '\n', '备选个数:', len(best), '\n',
          '最优和:', sum(best), '\n', '目标和:', tgt_sum, '\n',
          '差值:', tgt_sum-sum(best))
