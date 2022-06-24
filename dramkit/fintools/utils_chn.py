# -*- coding: utf-8 -*-

import os
import calendar
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from chinese_calendar import is_workday
from dramkit.gentools import isnull
from dramkit.gentools import sort_dict
from dramkit.iotools import load_csv
from dramkit import datetimetools as dttools


# 上市公司公告披露日期：https://zhuanlan.zhihu.com/p/29007967
# http://www.csrc.gov.cn/pub/zjhpublic/zjh/202103/t20210319_394491.htm
# https://xueqiu.com/5749132507/142221762
#https://zhuanlan.zhihu.com/p/127156379
# 一季报和三季报不强制披露
LAST_DATE_Q1 = '04-30' # 一季报披露时间（4月1日 – 4月30日）
LAST_DATE_SEMI = '08-31' # 半年报披露时间（7月1日 – 8月31日）
LAST_DATE_Q3 = '10-31' # 三季报披露时间（10月1日 – 10月31日）
LAST_DATE_ANN = '04-30' # 年报披露时间（1月1日 – 4月30日）


def get_finreport_date_by_delta(end_date, n, out_format='str'):
    '''
    | 给定一个报告期，获取往后推n期的报告期，若n为负，则往前推
    | 注：函数实际上计算的是end_date所在月往后推3*n季度之后所处月的最后一天，
      所以如果入参end_date不是财报日期所在月，则返回结果也不是财报日期
    '''
    end_date = pd.to_datetime(end_date)
    target_date = end_date + relativedelta(months=3*n)
    month_range = calendar.monthrange(target_date.year, target_date.month)
    target_date = target_date.year*10000 + target_date.month*100 + month_range[1]
    if out_format == 'int':
        return target_date
    elif out_format == 'str':
        return str(target_date)
    elif out_format == 'timestamp':
        target_date = pd.to_datetime(target_date, format='%Y%m%d')
    else:
        raise ValueError('out_format参数错误！')


def get_last_effect_finreport_dates(date,
                                    force_q1q3=False,
                                    wait_finished=False,
                                    semi_annual_only=False,
                                    annual_only=False,
                                    adjust_report=False):
    '''
    TODO
    ----
    - 完成adjust_report设置部分
    - 检查确认和完善***备注处


    | 获取距离date最近的财报日期，即date可以取到的最新财报数据对应的财报日期
    | force_q1q3: 一季报和三季报是否为强制披露，默认不强制披露
    | wait_finished: 是否必须等财报发布结束才可取数据
    | semi_annual_only: 仅考虑年报和中报
    | annual_only: 仅考虑年报
    | adjust_report: 是否考虑报告调整
    | 注：
    | 2020年特殊处理（2019年报和2020一季报不晚于20200630披露）
    | http://www.csrc.gov.cn/pub/zjhpublic/zjh/202004/t20200407_373381.htm
    '''

    assert not all([semi_annual_only, annual_only]), \
           'semi_annual_only和annual_only只能一个为True'

    def get_last_report_period(date):
        '''
        根据date获取最近一个报告期的year和season，将(year, season)编码成一个整数
        date须为整数型
        '''
        year = date // 10000 # 年份
        season = (date % 10000 // 100 - 1) // 3
        return 4 * year + season

    def from_report_period_to_date(report_period):
        '''
        将get_last_report_period生成的(year, season)整数编码还原成报告日期
        '''
        year = report_period // 4
        season = report_period % 4
        if season == 0:
            season += 4
            year -= 1
        if season == 1:
            return year * 10000 + 331
        elif season == 2:
            return year * 10000 + 630
        elif season == 3:
            return year * 10000 + 930
        elif season == 4:
            return year * 10000 + 1231
        else:
            raise ValueError('未识别的`period`：{}！'.format(report_period))

    # 日期转为整数
    if not isinstance(date, int):
        _, joiner = dttools.get_date_format(date)
        date = int(dttools.date_reformat(date, ''))
        int_ = False
    else:
        int_ = True
    year = date // 10000

    # 距离date最近的报告期
    last_period = get_last_report_period(date)
    n = 4 if (year == 2020 or wait_finished) else 3 # ***
    last_periods = [last_period - i for i in range(n)]
    last_dates = [from_report_period_to_date(x) for x in last_periods]

    # 不同报告日期的影响范围(下一个报告披露完成之前可能取到数据的最新报告日期)
    if year == 2020:
        if wait_finished:
            if not force_q1q3:
                effective_periods = {1231: [10701, 10831],
                                     331: [701, 831],
                                     630: [901, 10630],
                                     930: [1101, 10630]}
            else:
                effective_periods = {1231: [10701, 10831],
                                     331: [701, 831],
                                     630: [901, 1031],
                                     930: [1101, 10630]}
        else:
            if not force_q1q3:
                effective_periods = {1231: [10101, 10831],
                                     331: [401, 831],
                                     630: [701, 10630],
                                     930: [1001, 10630]}
            else:
                effective_periods = {1231: [10101, 10831],
                                     331: [401, 831],
                                     630: [701, 1031],
                                     930: [1001, 10630]}
    else:
        if wait_finished:
            if not force_q1q3:
                effective_periods = {1231: [10501, 10831],
                                     331: [501, 831],
                                     630: [901, 10430],
                                     930: [1101, 10430]}
            else:
                effective_periods = {1231: [10501, 10430],
                                     331: [501, 831],
                                     630: [901, 1031],
                                     930: [1101, 10430]}
        else:
            if not force_q1q3:
                effective_periods = {1231: [10101, 10831],
                                     331: [401, 831],
                                     630: [701, 10430],
                                     930: [1001, 10430]}
            else:
                effective_periods = {1231: [10101, 10430],
                                     331: [401, 831],
                                     630: [701, 1031],
                                     930: [1001, 10430]}

    dates = []
    for last_date in last_dates:
        last_month = last_date % 10000
        last_year = last_date // 10000 * 10000
        beg_, end_ = [last_year+x for x in effective_periods[last_month]]
        # print(last_date, beg_, end_)
        if date >= beg_ and date <= end_:
            dates.append(last_date)

    # 仅考虑年报中报
    if semi_annual_only:
        dates_ = {}
        for date in dates:
            month = date % 10000 // 100
            year = date // 10000
            if month in [12, 6]:
                dates_[date] = date
            elif month == 9:
                dates_[date] = year * 10000 + 630
            elif month == 3:
                dates_[date] = (year-1) * 10000 + 1231
        dates = dates_

    # 仅考虑年报
    if annual_only:
        dates_ = {}
        for date in dates:
            month = date % 10000 // 100
            year = date // 10000
            if month == 12:
                dates_[date] = date
            elif month < 12:
                dates_[date] = (year-1) * 10000 + 1231
        dates = dates_

    # # 是否调整
    # if adjust_report:
    #     # 前一年？
    #     if isinstance(dates, list):
    #         dates = {x: x-10000 for x in dates}
    #     elif isinstance(dates, dict):
    #         dates = {k: v - 10000 for k, v in dates.items()}
    #     # 前一期？


    if not int_:
        if isinstance(dates, list):
            dates = [dttools.date_reformat(str(x), joiner=joiner) for x in dates]
        elif isinstance(dates, dict):
            dates = {dttools.date_reformat(str(k), joiner=joiner):
                     dttools.date_reformat(str(v), joiner=joiner) \
                     for k, v in dates.items()}
                
    # 返回结果排序
    if isinstance(dates, list):
        dates.sort()
    else:
        dates = sort_dict(dates, by='value')

    return dates


def get_code_ext(code):
    '''
    TODO
    ----
    检查更新代码规则，增加北交所
    
    
    | 返回带交易所后缀的股票代码格式，如输入`300033`，返回`300033.SZ`
    | code目前可支持[A股、B股、50ETF期权、300ETF期权]，根据需要更新
    | 如不能确定后缀，则直接返回code原始值
    |
    | http://www.sse.com.cn/lawandrules/guide/jyznlc/jyzn/c/c_20191206_4960455.shtml
    '''

    code = str(code)

    # 上交所A股以'600'、'601'、'603'、'688'（科创板）开头，B股以'900'开头，共6位
    if len(code) == 6 and code[0:3] in ['600', '601', '603', '688', '900']:
        return code + '.SH'

    # 上交所50ETF期权和300ETF期权代码以'100'开头，共8位
    if len(code) == 8 and code[0:3] == '100':
        return code + '.SH'

    # 深交所A股以'000'（主板）、'002'（中小板）, '300'（创业板）开头，共6位
    # 深交所B股以'200'开头，共6位
    if len(code) == 6 and code[0:3] in ['000', '002', '300', '200']:
        return code + '.SZ'

    # 深交所300ETF期权代码以'900'开头，共8位
    if len(code) == 8 and code[0:3] == '900':
        return code + '.SZ'

    return code


def get_trade_fee_Astock(code, buy_or_sel, vol, price,
                         fee_least=5, fee_pct=2.5/10000):
    '''
    普通A股股票普通交易费用计算
    
    TODO
    ----
    - 检查更新交易费率变化(若同一个交易所不同板块费率不同，新增按板块计算)
    - 新增北交所
    '''
    if str(code)[0] == '6':
        return trade_fee_Astock('SH', buy_or_sel, vol, price, fee_least, fee_pct)
    else:
        return trade_fee_Astock('SZ', buy_or_sel, vol, price, fee_least, fee_pct)


def trade_fee_Astock(mkt, buy_or_sel, vol, price,
                     fee_least=5, fee_pct=2.5/10000):
    '''
    TODO
    ----
    - 检查更新交易费率变化(若同一个交易所不同板块费率不同，新增按板块计算)
    - 新增北交所
    
    
    普通A股股票普通交易费用计算

    Parameters
    ----------
    mkt : str
        'SH'('sh', 'SSE')或'SZ'('sz', 'SZSE')，分别代表上海和深圳市场
    buy_or_sel : str
        'B'('b', 'buy')或'S'('s', 'sell', 'sel')，分别标注买入或卖出
    vol : int
        量（股）
    price : float
        价格（元）
    fee_least : float
        券商手续费最低值
    fee_pct : float
        券商手续费比例

    Returns
    -------
    fee_all : float
        交易成本综合（包含交易所税费和券商手续费）


    | 收费标准源于沪深交易所官网，若有更新须更改：
    | http://www.sse.com.cn/services/tradingservice/charge/ssecharge/（2020年4月）
    | http://www.szse.cn/marketServices/deal/payFees/index.html（2020年2月）
    '''

    if mkt in ['SH', 'sh', 'SSE']:
        if buy_or_sel in ['B', 'b', 'buy']:
            tax_pct = 0.0 / 1000 # 印花税
            sup_pct = 0.2 / 10000 # 证券交易监管费
            hand_pct = 0.487 / 10000 # 经手（过户）费
        elif buy_or_sel in ['S', 's', 'sell', 'sel']:
            tax_pct = 1.0 / 1000
            sup_pct = 0.2 / 10000
            hand_pct = 0.487 / 10000

        net_cash = vol * price # 交易额
        fee_mkt = net_cash * (tax_pct + sup_pct + hand_pct) # 交易所收费
        fee_sec = max(fee_least, net_cash * fee_pct) # 券商收费

    if mkt in ['SZ', 'sz', 'SZSE']:
        if buy_or_sel in ['B', 'b', 'buy']:
            tax_pct = 0.0 / 1000 # 印花税
            sup_pct = 0.2 / 10000 # 证券交易监管费
            hand_pct = 0.487 / 10000 # 经手（过户）费
        elif buy_or_sel in ['S', 's', 'sell', 'sel']:
            tax_pct = 1.0 / 1000
            sup_pct = 0.2 / 10000
            hand_pct = 0.487 / 10000

        net_cash = vol * price # 交易额
        fee_mkt = net_cash * (tax_pct + sup_pct + hand_pct) # 交易所收费
        fee_sec = max(fee_least, net_cash * fee_pct) # 券商收费

    fee_all = fee_mkt + fee_sec

    return fee_all


def _is_trade_day_chncal(date=None):
    '''
    利用chinese_calendar库判断date（str格式）是否为交易日
    '''
    if isnull(date):
        date = dttools.today_date()
    date = dttools.date_reformat(date, '')
    date_dt = datetime.strptime(date, '%Y%m%d')
    return is_workday(date_dt) and date_dt.weekday() not in [5, 6]


def is_trade_day(date=None, trade_dates=None):
    '''判断date（str格式）是否为交易日'''
    if isnull(trade_dates):
        return _is_trade_day_chncal(date)
    if isnull(date):
        date = dttools.today_date()
    dates = get_trade_dates(date, date, trade_dates)
    if date in dates:
        return True
    return False


def _get_recent_trade_date_chncal(date=None, dirt='post'):
    '''
    | 若date为交易日，则直接返回date，否则返回下一个(dirt='post')或上一个(dirt='pre')交易日
    | 注：若chinese_calendar库统计的周内工作日与交易日有差异或没更新，可能导致结果不准确
    '''
    assert dirt in ['post', 'pre']
    if isnull(date):
        date = dttools.today_date()
    if dirt == 'post':
        while not _is_trade_day_chncal(date):
            date = dttools.date_add_nday(date, 1)
    elif dirt == 'pre':
        while not _is_trade_day_chncal(date):
            date = dttools.date_add_nday(date, -1)
    return date


def get_recent_trade_date(date=None, dirt='pre', trade_dates=None):
    '''
    | 若date为交易日，则直接返回date，否则返回下一个(dirt='post')或上一个(dirt='pre')交易日
    '''
    assert dirt in ['post', 'pre']
    if isnull(trade_dates):
        return _get_recent_trade_date_chncal(date, dirt)
    if is_trade_day(date, trade_dates):
        return date
    if dirt == 'pre':
        return get_next_nth_trade_date(date, -1, trade_dates)
    elif dirt == 'post':
        return get_next_nth_trade_date(date, 1, trade_dates)


def _get_next_nth_trade_date_chncal(date=None, n=1):
    '''
    | 给定日期date，返回其后第n个交易日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    '''
    if isnull(date):
        date = dttools.today_date()
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = dttools.date_add_nday(date, n_add)
        if _is_trade_day_chncal(date):
            tmp += 1
    return date


def get_next_nth_trade_date(date=None, n=1,
                            trade_dates_df_path=None):
    '''
    | 给定日期date，返回其后第n个交易日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    | trade_dates_df_path可以为历史交易日期数据存档路径，也可以为pd.DataFrame
    | 注：默认trade_dates_df_path数据格式为tushare格式：
    |     exchange,date,is_open
    |     SSE,2020-09-02,1
    |     SSE,2020-09-03,1
    |     SSE,2020-09-04,1
    |     SSE,2020-09-05,0
    '''
    
    if isnull(trade_dates_df_path):
        return _get_next_nth_trade_date_chncal(date=date, n=n)
    
    if isnull(date):
        if isnull(date):
            date, joiner = dttools.today_date('-'), '-'
    else:
        _, joiner = dttools.get_date_format(date)
        date = dttools.date_reformat(date, '-')
        
    if isinstance(trade_dates_df_path, str) and \
                        os.path.isfile(trade_dates_df_path):
        dates = load_csv(trade_dates_df_path)
    elif isinstance(trade_dates_df_path, pd.core.frame.DataFrame):
            dates = trade_dates_df_path.copy()
    else:
        raise ValueError('trade_dates_df_path不是pd.DataFrame或路径不存在！')

    dates.sort_values('date', ascending=True, inplace=True)
    dates.drop_duplicates(subset=['date'], keep='last', inplace=True)
    dates['tmp'] = dates[['date', 'is_open']].apply(lambda x:
                1 if x['is_open'] == 1 or x['date'] == date else 0, axis=1)
    dates = list(dates[dates['tmp'] == 1]['date'].unique())
    dates.sort()
    if not date in dates:
        return _get_next_nth_trade_date_chncal(date=date, n=n)
    else:
        idx = dates.index(date)
        if -1 < idx+n < len(dates):
            return dttools.date_reformat(dates[idx+n], joiner)
        else:
            return _get_next_nth_trade_date_chncal(date=date, n=n)


def _get_trade_dates_chncal(start_date, end_date=None):
    '''
    利用chinese_calendar库获取指定起止日期内的交易日期（周内的工作日）
    '''
    _, joiner = dttools.get_date_format(start_date)
    if isnull(end_date):
        end_date = dttools.today_date(joiner=joiner)
    dates = pd.date_range(start_date, end_date)
    dates = [x.strftime(joiner.join(['%Y', '%m', '%d'])) for x in dates if \
                        is_workday(x) and x.weekday() not in [5, 6]]
    dates = [dttools.date_reformat(x, joiner) for x in dates]
    return dates


def get_trade_dates(start_date, end_date=None,
                    trade_dates_df_path=None, joiner=2):
    '''
    | 获取起止日期之间(从start_date到end_date)的交易日期，返回列表
    | trade_dates_df_path可以为历史交易日期数据存档路径，也可以为pd.DataFrame
    | 注：默认trade_dates_df_path数据格式为tushare格式：
    |     exchange,date,is_open
    |     SSE,2020-09-02,1
    |     SSE,2020-09-03,1
    |     SSE,2020-09-04,1
    |     SSE,2020-09-05,0
    '''

    _, joiner1 = dttools.get_date_format(start_date)
    if isnull(end_date):
        end_date = dttools.today_date(joiner=joiner1)

    if isnull(trade_dates_df_path):
        return dttools.get_dates_between(start_date, end_date, keep1=True,
               keep2=True, only_workday=True, del_weekend=True, joiner=joiner)
    else:
        if isinstance(trade_dates_df_path, str) and \
                        os.path.isfile(trade_dates_df_path):
            data = load_csv(trade_dates_df_path)
        elif isinstance(trade_dates_df_path, pd.core.frame.DataFrame):
            data = trade_dates_df_path.copy()
        else:
            raise ValueError('trade_dates_df_path不是pd.DataFrame或路径不存在！')
        
        if data['date'].notna().sum() == 0:
            return dttools.get_dates_between(start_date, end_date, keep1=True,
                   keep2=True, only_workday=True, del_weekend=True, joiner=joiner)
        
        _, joiner2 = dttools.get_date_format(end_date)
        start_date = dttools.date_reformat(start_date, '-')
        end_date = dttools.date_reformat(end_date, '-')

        last_date = data['date'].max()
        data = data[data['date'] >= start_date]
        data = data[data['date'] <= end_date] 
        
        if joiner == 2:
            joiner = joiner2
        elif joiner == 1:
            joiner = joiner1
        dates = data[data['is_open'] == 1]['date']
        dates = [dttools.date_reformat(x, joiner) for x in list(dates)]
        dates.sort()

        if last_date == end_date:
            return dates
        else:
            start_date_ = dttools.date_add_nday(last_date, n=1)
            dates_ = dttools.get_dates_between(start_date_, end_date,
                                 keep1=True, keep2=True, only_workday=True,
                                 del_weekend=True, joiner=joiner)
            return dates + dates_


def get_num_trade_dates(start_date, end_date, trade_dates_df_path=None):
    '''给定起止时间，获取可交易天数'''
    if not isnull(trade_dates_df_path):
        trade_dates = get_trade_dates(start_date, end_date,
                                      trade_dates_df_path)
    else:
        trade_dates = _get_trade_dates_chncal(start_date, end_date)
    return len(trade_dates)
