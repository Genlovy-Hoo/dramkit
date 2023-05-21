# -*- coding: utf-8 -*-

import time
# from .find_addends_utils import tol2side_x_eq_y, backfind_sml1st_index
# # from find_addends_utils import tol2side_x_eq_y, backfind_sml1st_index
from dramkit.find_addends.find_addends_utils import tol2side_x_eq_y
from dramkit.find_addends.find_addends_utils import backfind_sml1st_index
from dramkit.logtools.utils_logger import logger_show


def find_addends_smlfirst(tgt_sum, alts, n_adds=None,
                          tol_below=0.0, tol_above=0.0,
                          max_loop=1000000, global_loop=False,
                          log_info=False, save_process=False,
                          logger=None):
    '''
    | 从给定的列表alts（可以是负数）中选取若干个加数之和最接近tgt_sum，小的备选数优先

    | 思路: 与 :func:`dramkit.find_addends.find_addends_bigfirst.find_addends_bigfirst` 类似
    |      只不过小的数优先，从而避免了有负数时找不到最优解的情况

    Parameters
    ----------
    tgt_sum : float, int
        目标和
    alts : list
        备选数列表
    n_adds : int
        限制备选加数个数为n_adds，默认无限制
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

    if len(alts) == 0:
        return [], [], adds_process

    alts.sort(reverse=True) # 降序（小的数优先进入）

    # 初始化，idx_last记录最新进入的数的索引号
    idx_last = len(alts) - 1
    chosed_idxs = [idx_last]
    chosed_addends = [alts[idx_last]]
    choseds_best = []
    if save_process:
        adds_process.append(chosed_addends.copy())
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

        # 无最优解（搜索到最大值且只剩它一个备选数），结束搜索
        if idx_last == 0 and len(chosed_idxs) == 1:
            if log_info:
                logger_show('无最优解（搜索到最大值且只剩它一个备选数）结束搜索。',
                            logger, 'info')
            break

        # 刚好搜索到最大值且不是最优解，此时去掉最大值并更改最大值前面一个进去的值
        if idx_last == 0:
            idx_last = chosed_idxs[-2]-1
            del chosed_idxs[-2:]
            chosed_idxs.append(idx_last)
            del chosed_addends[-2:]
            chosed_addends.append(alts[idx_last])
            if save_process:
                adds_process.append(chosed_addends.copy())
            now_sum = sum(chosed_addends)
            continue

        # 下一个备选数
        # idx_last -= 1
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
        logger_show('used time {}s.'.format(round(time.monotonic() - start_time,6)),
                    logger, 'info')

    return choseds_best, chosed_addends, adds_process


if __name__ == '__main__':
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

    alts = [1317876.0, 2193621.0, 3296537.0] + \
           [1201008.0, 1254715.0, 1269351.0, 1277352.0, 1291173.81, 1317600.0,
            1330352.0, 1353600.0, 1354412.0, 1370457.0, 1374766.0, 1408844.75,
            1439220.0, 1510600.0, 1524456.0, 1617486.0, 1672787.0, 1681834.0,
            1686519.0, 1687405.0, 1690206.0, 1695000.0, 1724995.0, 1751435.0,
            1785641.0, 1843309.36, 1856886.0, 1894970.0, 1898708.0, 1912052.0,
            1921500.0, 1929600.0, 1963981.2, 1969590.0, 2023032.0, 2026415.0,
            2196599.0, 2199462.0, 2199786.0, 2209606.0, 2209663.0, 2235424.0,
            2240000.0, 2243847.0, 2249822.0, 2258363.0, 2270664.28, 2322470.17,
            2324000.0, 2340800.0, 2416128.0, 2574972.0, 2727184.0, 2752425.0,
            2759907.72, 2852438.0, 2866005.1, 2868857.0, 2975417.2,
            3292897.0] + \
           [3298769.36, 3329329.0, 3433606.0, 3444547.1, 449625.1, 4036216.0,
            4415600.0, 4431336.72, 4637542.0, 4641028.0, 5442861.0, 5499746.0,
            5502825.0, 5505000.0, 5574654.0, 5636556.0, 5774073.0, 5788190.0,
            5810000.0, 9069410.0, 11291809.0, 11504962.0, 11590342.0,
            11618838.0, 11620000.0, 11620000.0] + \
           [-4036216.0, -2322470.17]
    tgt_sum = sum([1317876.0, 2193621.0, -4036216.0, -2322470.17])

    alts = [22, 15, 14, 13, 7, 6.1, 5, 21.5, 100]
    tgt_sum = 22 + 21 + 4.1

    # alts = [200, 107, 100, 99, 98, 6, 5, 3, -1, -20, -25]
    # tgt_sum = 100 + 6 - 25

    alts = [100, -100, -105, -102, -25, -30, -1]
    tgt_sum = -26

    n_adds = None
    tol_below = 0.0
    tol_above = 0.0
    max_loop = 1000000000
    global_loop = True
    log_info = True
    save_process = True
    logger = None


    best, choseds, mid = find_addends_smlfirst(tgt_sum, alts, n_adds=n_adds,
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
