# -*- coding: utf-8 -*-

from dramkit.gentools import TimeRecoder
from dramkit.find_addends.find_addends_bigfirst import find_addends_bigfirst as fab
from dramkit.find_addends.find_addends_utils import get_alts_sml
from dramkit.iotools import unpickle_file
from dramkit.speedup.multi_thread import multi_thread_threading
from dramkit.speedup.multi_process_concurrent import multi_process_concurrent


def find_addends_all(tgt_sums, alts):
    chosed_all = []
    for tgt_sum in tgt_sums:
        choseds, _, _ = fab(tgt_sum, alts, n_adds=None, check_alts=False,
                             add_num=None, tol_below=0.1, tol_above=0.1,
                             max_loop=10000000, global_loop=False,
                             log_info=False, save_process=False, logger=None)
        result = {'tgt_sum': tgt_sum,
                  'choseds': choseds,
                  'choseds_sum': sum(choseds),
                  'diff': tgt_sum - sum(choseds)}
        print(result)
        print('\n')
        chosed_all.append(result)
    return chosed_all


def find_addends_all_multi_thread(tgt_sums, alts):
    args_list = []
    for tgt_sum in tgt_sums:
        args_list.append([tgt_sum, alts, None, False, None, 0.1, 0.1,
                          10000000, False, False, False, None])
    results = multi_thread_threading(fab, args_list)
    chosed_all = []
    for k in range(len(tgt_sums)):
        result = {'tgt_sum': tgt_sums[k],
                  'choseds': results[k][0],
                  'choseds_sum': sum(results[k][0]),
                  'diff': tgt_sums[k] - sum(results[k][0])}
        print(result)
        print('\n')
        chosed_all.append(result)
    return chosed_all


def func_new(args):
    return fab(*args)


def find_addends_all_multi_process(tgt_sums, alts):
    args_list = []
    for tgt_sum in tgt_sums:
        args_list.append([tgt_sum, alts, None, False, None, 0.1, 0.1,
                          10000000, False, False, False, None])
    results = multi_process_concurrent(func_new, args_list)
    chosed_all = []
    for k in range(len(tgt_sums)):
        result = {'tgt_sum': tgt_sums[k],
                  'choseds': results[k][0],
                  'choseds_sum': sum(results[k][0]),
                  'diff': tgt_sums[k] - sum(results[k][0])}
        print(result)
        print('\n')
        chosed_all.append(result)
    return chosed_all


if __name__ == '__main__':
    alts1 = unpickle_file('./20190911_002751.npy')
    tgt_sums1 = [18397291.0, 8629911.0, 7182895.0, 6984261.0, 6426745.0]
    alts1 = get_alts_sml(sum(tgt_sums1), alts1, add_num=30)

    alts2 = unpickle_file('./20190909_000063.npy')
    tgt_sums2 = [269798749.0, 150380309.0, 133511499.0, 131281342.0,
                 123689533.0]
    alts2 = get_alts_sml(sum(tgt_sums2), alts2, add_num=30)


    tgt_sums, alts = tgt_sums1, alts1
    # tgt_sums, alts = tgt_sums2, alts2

    tr = TimeRecoder()
    choseds_all = find_addends_all(tgt_sums, alts)
    tr.used()

    tr = TimeRecoder()
    choseds_all_multi_thread = find_addends_all_multi_thread(tgt_sums, alts)
    tr.used()

    tr = TimeRecoder()
    choseds_all_multi_proces = find_addends_all_multi_process(tgt_sums, alts)
    tr.used()
