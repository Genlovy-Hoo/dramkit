# -*- coding: utf-8 -*-

from dramkit.iotools import unpickle_file
from dramkit.find_addends.find_addends_utils import get_alts_sml
from dramkit.find_addends.find_addends_bigfirst import find_addends_bigfirst as fab


def search_addends_2step(tgt_sum, alts, add_num_stp1=30, add_num_stp2=300,
                         tol = 0.50, tol_above=0.50, tol_less_stp1=1000000,
                         tol_less_stp2=0.50, max_loop=1000000, log_info=False):

    alts_stp1 = get_alts_sml(tgt_sum, alts, tol=tol_above,
                             add_num=add_num_stp1)
    choseds_stp1, _, _ = fab(tgt_sum, alts_stp1, tol_above=tol_above,
                             tol_below=tol_less_stp1, max_loop=max_loop,
                             log_info=log_info)
    if abs(tgt_sum-sum(choseds_stp1)) < tol:
        return choseds_stp1

    sum_stp2 = tgt_sum - sum(choseds_stp1)
    alts_stp2 = alts.copy()
    for x in choseds_stp1:
        alts_stp2.remove(x)
    alts_stp2 = get_alts_sml(sum_stp2, alts_stp2, tol=tol_above,
                             add_num=add_num_stp2)
    choseds_stp2, _, _ = fab(sum_stp2, alts_stp2, tol_above=tol_above,
                             tol_below=tol_less_stp2, max_loop=max_loop,
                             log_info=log_info)

    choseds_final = choseds_stp1 + choseds_stp2

    return choseds_final


if __name__ == '__main__':
    alts1 = unpickle_file('./20190911_002751.npy')
    tgt_sums1 = [18397291.0, 8629911.0, 7182895.0, 6984261.0, 6426745.0]

    alts2 = unpickle_file('./20190909_000063.npy')
    tgt_sums2 = [269798749.0, 150380309.0, 133511499.0, 131281342.0,
                 123689533.0]

    tgt_sums, alts = tgt_sums1, alts1
    # tgt_sums, alts = tgt_sums2, alts2

    add_num_stp1 = 50
    add_num_stp2 = 500
    tol = 0.50
    tol_above=0.50
    tol_less_stp1=1000000
    tol_less_stp2=0.50
    max_loop=10000000
    log_info=False

    for tgt_sum in tgt_sums:
        choseds=search_addends_2step(tgt_sum, alts,
                                     add_num_stp1=add_num_stp1,
                                     add_num_stp2=add_num_stp2,
                                     tol=tol, tol_above=tol_above,
                                     tol_less_stp1=tol_less_stp1,
                                     tol_less_stp2=tol_less_stp2,
                                     max_loop=max_loop, log_info=log_info)
        print(' choseds:', choseds, '\n', 'choseds num:', len(choseds), '\n',
              'choseds_sum:', sum(choseds), '\n', 'tgt_sum:', tgt_sum, '\n',
              'diff:', tgt_sum-sum(choseds), '\n')
