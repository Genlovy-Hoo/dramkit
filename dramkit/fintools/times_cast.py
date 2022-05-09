# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#%%
class TimesCasterLottery(object):
    '''
    | 简单倍投法博弈过程计算
    | 适用场景：
    | 单次博弈仅有两种结果，即要么盈利，要么本金亏完，且每次投入只能为基准投入得整数倍
     （例如彩票，要么中奖，要么本金亏完，每次只能买整数张）
    '''

    def __init__(self, base_cost=2, base_back=5, expect_gain_pct=50/100,
                 init_n=1):
        '''
        | base_cost: 单次博弈成本（比如彩票2块钱一张）
        | base_back: 单次胜出获得收入（比如彩票中奖最低奖励为5块钱）
        | expect_gain_pct: 倍投一个回合结束时期望盈利占总投入的比例
        | init_n: 初始投入的基准倍数（比如init_n=2表示第一次买两张彩票）
        '''

        if base_back <= base_cost:
            raise ValueError('单次胜出回报base_back必须大于单次成本base_cost！')

        self.base_cost = base_cost
        self.base_back = base_back
        self.init_n = init_n
        self.expect_gain_pct = expect_gain_pct

    def _get_next_new_cost(self, pre_total_cost, pre_total_n):
        '''
        给定截止上一次的总投入，计算需要新投入的新成本
        '''

        fenzi = pre_total_n * self.base_cost * (1 + self.expect_gain_pct)
        fenmu = self.base_back - self.base_cost * (1 + self.expect_gain_pct)
        next_new_n = np.ceil(fenzi / fenmu)
        next_new_cost = next_new_n * self.base_cost

        return next_new_n, next_new_cost

    def play_n_round(self, n):
        '''
        假设玩n次之后实现盈利，计算博弈过程
        '''

        results = [] # 保存博弈过程

        # 首次投注
        new_n = self.init_n
        new_cost = new_n * self.base_cost
        total_n = new_n
        total_cost = new_cost
        back_if_over = new_n * self.base_back
        gain_if_over = back_if_over - total_cost
        results.append([1, new_n, new_cost, total_n, total_cost, back_if_over,
                        gain_if_over])

        # 后续投注
        for k in range(0, n-1):
            new_n, new_cost = self._get_next_new_cost(total_cost, total_n)
            total_n += new_n
            total_cost += new_cost
            back_if_over = new_n * self.base_back
            gain_if_over = back_if_over - total_cost
            results.append([k+2, new_n, new_cost, total_n, total_cost,
                            back_if_over, gain_if_over])

        # 结果整理
        results = pd.DataFrame(results, columns=['序号', '投入量', '投入额',
                               '总投入量', '总投入额', '获胜结余', '盈利'])
        results['盈利%'] = (results['盈利'] / results['总投入额']) * 100
        results['盈利%'] = results['盈利%'].apply(
                                            lambda x: str(round(x, 4)) + '%')

        return results

#%%
class TimesCasterConFuture(object):
    '''
    | 永续合约交易倍投法交易博弈过程计算
    | 使用情景：永续合约，分买方-做多和卖方-做空两种角色（如比特币永续合约）
    '''

    def __init__(self, player, start_price, pct_add=2.0/100, lever=100,
                 expect_gain_pct=2.0/100, func_vol_add='ctrl_loss_pct_1.0/100',
                 init_vol=1, vol_dot=2, max_hold_rate=10/100, fee_pct=0.1/100,
                 n_future2target=0.001, seller_cost_base='open'):
        '''
        todo
        ----
        - 增加更多加仓方式选择（如每次加仓量跟当前加仓总次数相关联等）
        - fee卖出和买入分开考虑

        Parameters
        ----------
        player : str
            | 交易角色
            | - 买方：['buyer', 'Buyer', 'b', 'B', 'buy', 'Buy']
            | - 卖方：['seller', 'Seller', 's', 'S', 'sell', 'Sell', 'sel', 'Sel']
        start_price : float
            开始时（一个单位标的证券对应的）合约价格
            （eg. 一个BTC对应的永续合约价格为50000，注意：不是BTC的价格）
        pct_add : float
            合约价格每次跌|涨幅达到pct_add就加仓
        lever : int
            杠杆倍数
        expect_gain_pct : float
            期望盈利比例（注：成本为实际参与交易的资金总额，不是账户总额）
        func_vol_add : str
            | 加仓时的交易量确定方式
            | - 当func_vol_add指定为'base_x'时，每次加仓量为底仓init_vol的x倍
            | - 当func_vol_add指定为'hold_x'时，每次加仓量为持有量的x倍
            | - 当func_vol_add为'ctrl_loss_pct_x': 每次加仓量根据加仓完成之后的目标亏损比例
                来计算（计算目标成本时暂不考虑平仓时的交易成本），
                例如'ctrl_loss_pct_2.0/100'表示加仓完成之后亏损控制在2%
            | - 当func_vol_add指定为'base_kth_x'时，第k次加仓量为底仓init_vol的k倍乘以x
            | - 当func_vol_add指定为'hold_kth_x'时，第k次加仓量为底仓init_vol的k倍乘以x
        init_vol : int, float
            初次下单合约张数（基准开仓量）
        vol_dot : int
            下单量最多保留的小数点位数
        max_hold_rate : float
            最大仓位比例限制（用于计算账户所需保证金最低额）
        fee_pct : float
            交易成本（手续费）比例（单次）
        n_future2target : float
            一张合约对应多少个标的证券
            （用于计算保证金，eg. 一张BTC永续合约对应0.001个BTC）
        seller_cost_base : str
            | 计算卖方-做空盈亏比时成本用开仓价`open`还是平仓价`close`：
            | - 成本以open即卖出价格为准时有：卖出（开仓）价 * (1 - 盈亏%) = 买入（平仓）价
            | - 成本以close即买入价格为准时有：买入（平仓）价 * ( 1 + 盈亏%) = 卖出（开仓）价
        '''

        b_name = ['buyer', 'Buyer', 'b', 'B', 'buy', 'Buy']
        s_name = ['seller', 'Seller', 'seler', 'Seler' 's', 'S', 'sell',
                  'Sell', 'sel', 'Sel']
        if player in b_name:
            self.player = 'buyer'
        elif player in s_name:
            self.player = 'seller'
        else:
            raise ValueError('交易角色`player`设置错误，请检查！')

        self.start_price = start_price
        self.pct_add = pct_add
        self.lever = lever
        self.expect_gain_pct = expect_gain_pct
        self.init_vol = init_vol
        self.vol_dot = vol_dot
        self.max_hold_rate = max_hold_rate
        self.fee_pct = fee_pct
        self.n_future2target = n_future2target
        if seller_cost_base not in ['open', 'close']:
            raise ValueError('`seller_cost_base`必须为`open`或`close`！')
        self.seller_cost_base = seller_cost_base

        if 'ctrl_loss_pct_' in func_vol_add:
            ctrl_loss_pct = eval(func_vol_add.split('pct_')[-1])
            self.ctrl_loss_pct = ctrl_loss_pct
            self.F_add_vol = self._get_add_info_by_ctrl_loss
        elif 'base_kth_' in func_vol_add:
            self.multer_add_base = int(func_vol_add.split('base_kth_')[-1])
            self.F_add_vol = self._get_add_info_by_base_vol_kth
        elif 'hold_kth_' in func_vol_add:
            self.multer_add_hold = int(func_vol_add.split('hold_kth_')[-1])
            self.F_add_vol = self._get_add_info_by_hold_vol_kth
        elif 'base_' in func_vol_add:
            self.multer_add_base = int(func_vol_add.split('base_')[-1])
            self.F_add_vol = self._get_add_info_by_base_vol
        elif 'hold_' in func_vol_add:
            self.multer_add_hold = int(func_vol_add.split('hold_')[-1])
            self.F_add_vol = self._get_add_info_by_hold_vol

    def _get_Punit(self, price):
        '''根据当前单位标的证券对应的合约价格price计算一张合约的价格（所需保证金）'''
        return price * self.n_future2target / self.lever

    def _get_price(self, Punit):
        '''根据一张合约的价格（所需保证金）计算单位标的证券对应的合约价格price'''
        return Punit / (self.n_future2target / self.lever)

    def _get_endPunit_gain(self, Pcost):
        '''给定成本价Pcost（已经包含交易成本）计算平仓价以及平仓盈亏（一张合约）'''
        if self.player == 'buyer':
            endPunit = Pcost * (1 + self.expect_gain_pct) / (1 - self.fee_pct)
            gain = endPunit * (1 - self.fee_pct) - Pcost
            gain_pct = 100 * (gain / Pcost)
        elif self.player == 'seller':
            if self.seller_cost_base == 'open':
                # 计算盈亏比时成本使用开仓价格
                # 买入价*(1+self.fee_pct) = 卖出价*(1-self.expect_gain_pct)
                endPunit = Pcost * (1 - self.expect_gain_pct) / \
                                                            (1 + self.fee_pct)
                gain = Pcost - endPunit * (1 + self.fee_pct)
                gain_pct = 100 * (gain / Pcost)
            elif self.seller_cost_base == 'close':
                # 计算盈亏比时成本使用平仓价格
                # 买入价*(1+self.fee_pct)*(1+self.expect_gain_pct) = 卖出价
                endPunit = Pcost / (1 + self.expect_gain_pct) / \
                                                            (1 + self.fee_pct)
                gain = Pcost - endPunit * (1 + self.fee_pct)
                gain_pct = 100 * (gain / (endPunit * (1 + self.fee_pct)))
        return endPunit, gain, gain_pct

    def _get_add_hold_info(self, vol, Punit, pre_total_vol, pre_total_cost):
        '''
        | 根据加仓量vol、加仓价格Punit、持有总成本pre_total_cost和总量pre_total_vol，
        | 计算加仓后的持仓信息（总量, 总成本, 平均成本, 总价值）
        '''
        if self.player == 'buyer':
            total_vol = vol + pre_total_vol
            # 加仓后持仓总成本
            total_cost = pre_total_cost + vol * Punit * (1 + self.fee_pct)
            mean_cost = total_cost / total_vol # 加仓后平均成本
            total_val = total_vol * Punit # 加仓后持仓总价值（等于所需保证金）
        elif self.player == 'seller':
            total_vol = vol + pre_total_vol
            total_cost = pre_total_cost + vol * Punit * (1 - self.fee_pct)
            mean_cost = total_cost / total_vol
            total_val = total_vol * Punit
        return total_vol, total_cost, mean_cost, total_val

    def _get_add_info_by_base_vol_kth(self, Punit, pre_total_vol,
                                      pre_total_cost, k):
        '''
        | 计算加仓量并更新加仓后的持仓信息
        | 加仓量确定方式：基准开仓量的k倍再乘以一个常数
        | Punit: 当前合约价格（一张合约）
        | pre_total_vol: 当前（加仓前）持有总量
        | pre_total_cost: 当前（加仓前）持有总成本
        '''
        vol = self.multer_add_base * k * self.init_vol
        total_vol, total_cost, mean_cost, total_val = self._get_add_hold_info(
                                    vol, Punit, pre_total_vol, pre_total_cost)
        return vol, total_vol, total_cost, mean_cost, total_val

    def _get_add_info_by_hold_vol_kth(self, Punit, pre_total_vol,
                                      pre_total_cost, k):
        '''
        | 计算加仓量并更新加仓后的持仓信息
        | 加仓量确定方式：持仓量的k倍再乘以一个常数
        | Punit: 当前合约价格（一张合约）
        | pre_total_vol: 当前（加仓前）持有总量
        | pre_total_cost: 当前（加仓前）持有总成本
        '''
        vol = self.multer_add_hold * k * pre_total_vol
        total_vol, total_cost, mean_cost, total_val = self._get_add_hold_info(
                                    vol, Punit, pre_total_vol, pre_total_cost)
        return vol, total_vol, total_cost, mean_cost, total_val

    def _get_add_info_by_base_vol(self, Punit, pre_total_vol, pre_total_cost,
                                  **kwargs):
        '''
        | 计算加仓量并更新加仓后的持仓信息
        | 加仓量确定方式：基准开仓量的倍数
        | Punit: 当前合约价格（一张合约）
        | pre_total_vol: 当前（加仓前）持有总量
        | pre_total_cost: 当前（加仓前）持有总成本
        '''
        vol = self.multer_add_base * self.init_vol
        total_vol, total_cost, mean_cost, total_val = self._get_add_hold_info(
                                    vol, Punit, pre_total_vol, pre_total_cost)
        return vol, total_vol, total_cost, mean_cost, total_val

    def _get_add_info_by_hold_vol(self, Punit, pre_total_vol, pre_total_cost,
                                  **kwargs):
        '''
        | 计算加仓量并更新加仓后的持仓信息
        | 加仓量确定方式：持仓量的倍数
        | Punit: 当前合约价格（一张合约）
        | pre_total_vol: 当前（加仓前）持有总量
        | pre_total_cost: 当前（加仓前）持有总成本
        '''
        vol = self.multer_add_hold * pre_total_vol
        total_vol, total_cost, mean_cost, total_val = self._get_add_hold_info(
                                    vol, Punit, pre_total_vol, pre_total_cost)
        return vol, total_vol, total_cost, mean_cost, total_val

    def _get_add_info_by_ctrl_loss(self, Punit, pre_total_vol, pre_total_cost,
                                   **kwargs):
        '''
        | 计算加仓量并更新加仓后的持仓信息
        | 加仓量确定方式：根据目标损失得到（计算目标成本价时不考虑平仓手续费）
        | Punit: 当前合约价格（一张合约）
        | pre_total_vol: 当前（加仓前）持有总量
        | pre_total_cost: 当前（加仓前）持有总成本
        '''

        if self.player == 'buyer':
            # 买入价 * (1 - self.ctrl_loss_pct) = 卖出价
            target_cost = Punit / (1 - self.ctrl_loss_pct) # 加仓完成后的目标成本

            fenzi = pre_total_cost - target_cost * pre_total_vol
            fenmu = target_cost - Punit * (1 + self.fee_pct)
            vol = fenzi / fenmu
            vol = round(vol, self.vol_dot) # 加仓量
            vol = max(0, vol)

        elif self.player == 'seller':
            if self.seller_cost_base == 'open':
                # 卖出价 * (1 + self.ctrl_loss_pct) = 买入价
                target_cost = Punit / (1 + self.ctrl_loss_pct)
            elif self.seller_cost_base == 'close':
                # 卖出价 = 买入价 * (1 - self.ctrl_loss_pct)
                target_cost = Punit * (1 - self.ctrl_loss_pct)

            fenzi = pre_total_cost - target_cost * pre_total_vol
            fenmu = target_cost - Punit * (1 - self.fee_pct)
            vol = fenzi / fenmu
            vol = round(vol, self.vol_dot)
            vol = max(0, vol)

        total_vol, total_cost, mean_cost, total_val = self._get_add_hold_info(
                                    vol, Punit, pre_total_vol, pre_total_cost)

        return vol, total_vol, total_cost, mean_cost, total_val

    def play_n_round(self, N):
        '''
        假设开（加）仓N次之后锁仓，计算博弈过程
        '''

        results = [] # 保存博弈过程
        cols = ['序号', '成交价', '成交量', '持仓量', '总成本', '平均成本',
                '总价值（持仓保证金）', '盈亏', '盈亏%', '平仓价', '平-现价%',
                '平仓盈利', '平仓盈利%']

        # 首次开仓
        Punit0 = self._get_Punit(self.start_price) # 一张合约价格
        Punit = Punit0
        vol = self.init_vol
        total_vol = vol
        if self.player == 'buyer':
            total_cost = total_vol * Punit * (1 + self.fee_pct)
        elif self.player == 'seller':
            total_cost = total_vol * Punit * (1 - self.fee_pct)
        mean_cost = total_cost / total_vol
        total_val = total_vol * Punit
        endPunit, gain_unit_end, gain_pct_end = \
                                             self._get_endPunit_gain(mean_cost)
        if self.player == 'buyer':
            gain_val = total_val - total_cost # 没考虑平仓手续费
            gain_pct = 100 * (gain_val / total_cost)
        elif self.player == 'seller':
            gain_val = total_cost - total_val # 没考虑平仓手续费
            if self.seller_cost_base == 'open':
                gain_pct = 100 * (gain_val / total_cost)
            elif self.seller_cost_base == 'close':
                gain_pct = 100 * (gain_val / total_val)
        pct_Pend = 100 * (endPunit / Punit - 1)
        gain_val_end = gain_unit_end * total_vol
        results.append([1, Punit, vol, total_vol, total_cost, mean_cost,
                        total_val, gain_val, gain_pct, endPunit, pct_Pend,
                        gain_val_end, gain_pct_end])

        # 后续加仓
        for k in range(1, N):
            if self.player == 'buyer':
                # Punit = Punit * (1 - self.pct_add)
                Punit = Punit0 * (1 - self.pct_add * k)
            elif self.player == 'seller':
                # Punit = Punit * (1 + self.pct_add)
                Punit = Punit0 * (1 + self.pct_add * k)
            vol, total_vol, total_cost, mean_cost, total_val = \
                            self.F_add_vol(Punit, total_vol, total_cost, k=k)
            endPunit, gain_unit_end, gain_pct_end = \
                                             self._get_endPunit_gain(mean_cost)
            if self.player == 'buyer':
                gain_val = total_val - total_cost # 没考虑平仓手续费
                gain_pct = 100 * (gain_val / total_cost)
            elif self.player == 'seller':
                gain_val = total_cost - total_val # 没考虑平仓手续费
                if self.seller_cost_base == 'open':
                    gain_pct = 100 * (gain_val / total_cost)
                elif self.seller_cost_base == 'close':
                    gain_pct = 100 * (gain_val / total_val)
            pct_Pend = 100 * (endPunit / Punit - 1)
            gain_val_end = gain_unit_end * total_vol
            results.append([k+1, Punit, vol, total_vol, total_cost, mean_cost,
                        total_val, gain_val, gain_pct, endPunit, pct_Pend,
                        gain_val_end, gain_pct_end])

        # 结果整理
        results = pd.DataFrame(results, columns=cols)
        results['平仓盈利（杠）'] = results['平仓盈利'] * self.lever
        results['平仓盈利（杠）%'] = results['平仓盈利%'] * self.lever
        results['安全保证金（仓控）'] = results['总价值（持仓保证金）'] * \
                                                    (1 / self.max_hold_rate)
        results = results.reindex(columns=['序号', '成交价', '成交量', '持仓量',
                  '平均成本', '总成本', '总价值（持仓保证金）', '安全保证金（仓控）',
                  '盈亏', '盈亏%', '平仓价', '平仓盈利（杠）', '平仓盈利（杠）%',
                  '平-现价%'])
        results_unit = results.copy()
        results['成交价'] = results['成交价'].apply(lambda x: self._get_price(x))
        results['平仓价'] = results['平仓价'].apply(lambda x: self._get_price(x))
        results['开仓（成本）均价'] = results['平均成本'].apply(lambda x:
                                                    self._get_price(x))
        results = results.reindex(columns=['序号', '成交价', '成交量', '持仓量',
                  '开仓（成本）均价', '平均成本', '总成本', '总价值（持仓保证金）',
                  '安全保证金（仓控）', '盈亏', '盈亏%', '平仓价', '平仓盈利（杠）',
                  '平仓盈利（杠）%', '平-现价%'])

        return results, results_unit

#%%
if __name__ == '__main__':
    import time

    strt_tm = time.time()

    #%%
    # 彩票
    tc_lottery = TimesCasterLottery(base_cost=2, base_back=5,
                                    expect_gain_pct=50/100, init_n=1)
    results_lottery = tc_lottery.play_n_round(n=30)

    #%%
    # BTC永续合约（火币）
    start_price = 100000
    pct_add = 5.0/100
    expect_gain_pct = 2.0/100
    func_vol_add = 'ctrl_loss_pct_2.0/100'
    # func_vol_add = 'base_kth_3'
    # func_vol_add = 'hold_1'
    init_vol = 5.0
    n = 10

    # BTC永续-做多
    tcBTC_buy = TimesCasterConFuture('buyer', start_price,
                                     pct_add=pct_add,
                                     expect_gain_pct=expect_gain_pct,
                                     func_vol_add=func_vol_add,
                                     init_vol=init_vol)
    resultsBTC_buy, resultsBTC_buy_u = tcBTC_buy.play_n_round(n)

    # BTC永续-做空
    tcBTC_sel_open = TimesCasterConFuture('seller', start_price,
                                          pct_add=pct_add,
                                          expect_gain_pct=expect_gain_pct,
                                          func_vol_add=func_vol_add,
                                          init_vol=init_vol,
                                          seller_cost_base='open')
    resultsBTC_sel_open, resultsBTC_sel_open_u = \
                                            tcBTC_sel_open.play_n_round(n)

    tcBTC_sel_close = TimesCasterConFuture('seller', start_price,
                                           pct_add=pct_add,
                                           expect_gain_pct=expect_gain_pct,
                                           func_vol_add=func_vol_add,
                                           init_vol=init_vol,
                                           seller_cost_base='close')
    resultsBTC_sel_close, resultsBTC_sel_close_u = \
                                            tcBTC_sel_close.play_n_round(n)
                                            
    #%%
    # ETH永续合约（币安）
    start_price = 3000
    pct_add = 5.0/100
    lever = 50
    expect_gain_pct = 2.0/100
    func_vol_add = 'ctrl_loss_pct_2.0/100'
    # func_vol_add = 'base_kth_3'
    # func_vol_add = 'hold_1'
    init_vol = 0.1
    n = 10
    n_future2target = 1

    # ETH永续-做多
    tcETH_buy = TimesCasterConFuture('buyer', start_price,
                                     pct_add=pct_add,
                                     lever=lever,
                                     expect_gain_pct=expect_gain_pct,
                                     func_vol_add=func_vol_add,
                                     init_vol=init_vol,
                                     n_future2target=n_future2target)
    resultsETH_buy, resultsETH_buy_u = tcETH_buy.play_n_round(n)

    # ETH永续-做空
    tcETH_sel_open = TimesCasterConFuture('seller', start_price,
                                          pct_add=pct_add,
                                          lever=lever,
                                          expect_gain_pct=expect_gain_pct,
                                          func_vol_add=func_vol_add,
                                          init_vol=init_vol,
                                          n_future2target=n_future2target,
                                          seller_cost_base='open')
    resultsETH_sel_open, resultsETH_sel_open_u = \
                                            tcETH_sel_open.play_n_round(n)

    tcETH_sel_close = TimesCasterConFuture('seller', start_price,
                                           pct_add=pct_add,
                                           lever=lever,
                                           expect_gain_pct=expect_gain_pct,
                                           func_vol_add=func_vol_add,
                                           init_vol=init_vol,
                                           n_future2target=n_future2target,
                                           seller_cost_base='close')
    resultsETH_sel_close, resultsETH_sel_close_u = \
                                            tcETH_sel_close.play_n_round(n)

    #%%
    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
