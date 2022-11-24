# -*- coding: utf-8 -*-

# TODO
# 沪深A股日历接入tushare
# 判断农历月大月小，判断是否闰月等
# 增加八字排盘算命等


from __future__ import absolute_import, unicode_literals

import time
import datetime
import pandas as pd

from chncal.constants import (holidays,
                              in_lieu_days,
                              workdays)
from chncal.solar_terms import (SolarTerms,
                                SOLAR_TERMS_C_NUMS,
                                SOLAR_TERMS_MONTH,
                                SOLAR_TERMS_DELTA)

from chncal.constants_trade_dates import trade_dates
from chncal.constants_hko import gen_lun, lun_gen, gen_gz
from chncal.constants_fate import w_year, w_month, w_date, w_hour, song
from chncal.constants_zodiac_marry import zodiac_match
from chncal.constants_wuxing import tgwx, dzwx, tgdznywx


# # 干支纪年https://baike.baidu.com/item/干支纪年/3383226
# # 天干
# TG = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
# # 地支
# DZ = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
# 属相
SX = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
# k1, k2 = 0, 0
# TGDZ = [TG[k1]+DZ[k2]+'('+SX[k2]+')']
# while not (k1 == len(TG)-1 and  k2 == len(DZ)-1):
#     if k1 < len(TG)-1:
#         k1 += 1
#     else:
#         k1 = 0
#     if k2 < len(DZ)-1:
#         k2 += 1
#     else:
#         k2 = 0
#     TGDZ.append(TG[k1]+DZ[k2]+'('+SX[k2]+')')

TGDZ = ['甲子(鼠)', '乙丑(牛)', '丙寅(虎)', '丁卯(兔)',
        '戊辰(龙)', '己巳(蛇)', '庚午(马)', '辛未(羊)',
        '壬申(猴)', '癸酉(鸡)', '甲戌(狗)', '乙亥(猪)',
        '丙子(鼠)', '丁丑(牛)', '戊寅(虎)', '己卯(兔)',
        '庚辰(龙)', '辛巳(蛇)', '壬午(马)', '癸未(羊)',
        '甲申(猴)', '乙酉(鸡)', '丙戌(狗)', '丁亥(猪)',
        '戊子(鼠)', '己丑(牛)', '庚寅(虎)', '辛卯(兔)',
        '壬辰(龙)', '癸巳(蛇)', '甲午(马)', '乙未(羊)',
        '丙申(猴)', '丁酉(鸡)', '戊戌(狗)', '己亥(猪)',
        '庚子(鼠)', '辛丑(牛)', '壬寅(虎)', '癸卯(兔)',
        '甲辰(龙)', '乙巳(蛇)', '丙午(马)', '丁未(羊)',
        '戊申(猴)', '己酉(鸡)', '庚戌(狗)', '辛亥(猪)',
        '壬子(鼠)', '癸丑(牛)', '甲寅(虎)', '乙卯(兔)',
        '丙辰(龙)', '丁巳(蛇)', '戊午(马)', '己未(羊)',
        '庚申(猴)', '辛酉(鸡)', '壬戌(狗)', '癸亥(猪)']

# 农历2018年六月十二（公历2022.07.10）是甲子日
TGDZ_BASE_DATE = pd.to_datetime('2022.07.10')

# 公历2022.08.24凌晨是甲子时
# TGDZ_BASE_TIME = pd.to_datetime('2022.08.23 23:17:05')
TGDZ_BASE_TIME = pd.to_datetime('2022.08.23 23:00:00')

# 支持查交易日历的交易所和最早日期
# TODO
# 目前交易所首个交易日以tushare交易日历数据最早的日期为准，待查证准确的日期
MARKETS = {
            'SSE': datetime.date(1990, 12, 19), # 上交所
            'SZSE': datetime.date(1991, 7, 3), # 深交所
            'CFFEX': datetime.date(2006, 9, 8), # 中金所
            'SHFE': datetime.date(1991, 5, 28), # 上期所
            'CZCE': datetime.date(1990, 10, 12), # 郑商所
            'DCE': datetime.date(1993, 3, 1), # 大商所
            'INE': datetime.date(2017, 5, 23), # 上能源
        }


def _trans_date(date):
    if pd.isnull(date):
        date = datetime.datetime.now()
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif isinstance(date, int):
        date = pd.to_datetime(str(date))
    elif isinstance(date, time.struct_time):
        date = pd.to_datetime(
               datetime.datetime.fromtimestamp(time.mktime(date)))
    return date


def _wrap_date(date):
    '''
    transform datetime.datetime into datetime.date

    :type date: datetime.date | datetime.datetime
    :rtype: datetime.date
    '''
    date = _trans_date(date)
    if isinstance(date, datetime.datetime):
        date = date.date()
    return date


def _validate_date(*dates):
    '''
    check if the date(s) is supported

    :type date: datetime.date | datetime.datet'ime
    :rtype: datetime.date | list[datetime.date]
    '''
    if len(dates) != 1:
        return list(map(_validate_date, dates))
    date = _wrap_date(dates[0])
    if not isinstance(date, datetime.date):
        raise NotImplementedError('unsupported type {}, expected type is datetime.date'.format(type(date)))
    min_year, max_year = min(holidays.keys()).year, max(holidays.keys()).year
    if not (min_year <= date.year <= max_year):
        raise NotImplementedError(
            'no available data for year {}, only year between [{}, {}] supported'.format(date.year, min_year, max_year)
        )
    return date


def is_holiday(date=None):
    '''
    check if one date is holiday in China.
    in other words, Chinese people get rest at that day.

    :type date: datetime.date | datetime.datetime
    :rtype: bool
    '''
    return not is_workday(date)


def is_workday(date=None):
    '''
    check if one date is workday in China.
    in other words, Chinese people works at that day.

    :type date: datetime.date | datetime.datetime
    :rtype: bool
    '''
    date = _validate_date(date)
    weekday = date.weekday()
    return bool(date in workdays.keys() or (weekday <= 4 and date not in holidays.keys()))


def is_in_lieu(date=None):
    '''
    check if one date is in lieu in China.
    in other words, Chinese people get rest at that day because of legal holiday.

    :type date: datetime.date | datetime.datetime
    :rtype: bool
    '''
    date = _validate_date(date)
    return date in in_lieu_days


def get_holiday_detail(date=None):
    '''
    check if one date is holiday in China,
    and return the holiday name (None if it's a normal day)

    :type date: datetime.date | datetime.datetime
    :return: holiday bool indicator, and holiday name if it's holiday related day
    :rtype: (bool, str | None)
    '''
    date = _validate_date(date)
    if date in workdays.keys():
        return False, workdays[date]
    elif date in holidays.keys():
        return True, holidays[date]
    else:
        return date.weekday() > 4, None


def get_dates(start, end=None):
    '''
    get dates between start date and end date. (includes start date and end date)

    :type start: datetime.date | datetime.datetime
    :type end:  datetime.date | datetime.datetime
    :rtype: list[datetime.date]
    '''
    start, end = map(_wrap_date, (start, end))
    delta_days = (end - start).days
    return [start + datetime.timedelta(days=delta) for delta in range(delta_days + 1)]


def get_holidays(start, end=None, include_weekends=True):
    '''
    get holidays between start date and end date. (includes start date and end date)

    :type start: datetime.date | datetime.datetime
    :type end:  datetime.date | datetime.datetime
    :type include_weekends: bool
    :param include_weekends: False for excluding Saturdays and Sundays
    :rtype: list[datetime.date]
    '''
    start, end = _validate_date(start, end)
    if include_weekends:
        return list(filter(is_holiday, get_dates(start, end)))
    return list(filter(lambda x: x in holidays, get_dates(start, end)))


def get_workdays(start, end=None):
    '''
    get workdays between start date and end date. (includes start date and end date)

    :type start: datetime.date | datetime.datetime
    :type end:  datetime.date | datetime.datetime
    :rtype: list[datetime.date]
    '''
    start, end = _validate_date(start, end)
    return list(filter(is_workday, get_dates(start, end)))


def find_workday(delta_days=0, date=None):
    '''
    find the workday after {delta_days} days.

    :type delta_days: int
    :param delta_days: 0 means next workday (includes today), -1 means previous workday.
    :type date: datetime.date | datetime.datetime
    :param: the start point
    :rtype: datetime.date
    '''
    date = _wrap_date(date or datetime.date.today())
    if delta_days >= 0:
        delta_days += 1
    sign = 1 if delta_days >= 0 else -1
    for i in range(abs(delta_days)):
        if delta_days < 0 or i:
            date += datetime.timedelta(days=sign)
        while not is_workday(date):
            date += datetime.timedelta(days=sign)
    return date


def get_solar_terms(start=None, end=None):
    '''
    生成24节气
    通用寿星公式：https://www.jianshu.com/p/1f814c6bb475

    通式寿星公式：[Y×D+C]-L
    []里面取整数；Y=年数的后2位数；D=0.2422；L=Y/4，小寒、大寒、立春、雨水的 L=(Y-1)/4

    :type start: datetime.date
    :param start: 开始日期
    :type end: datetime.date
    :param end: 结束日期
    :rtype: list[(datetime.date, str)]
    '''
    start = _wrap_date(start)
    end = _wrap_date(end)
    if not 1900 <= start.year <= 2100 or not 1900 <= end.year <= 2100:
        raise NotImplementedError('only year between [1900, 2100] supported')
    D = 0.2422
    result = []
    year, month = start.year, start.month
    while year < end.year or (year == end.year and month <= end.month):
        # 按月计算节气
        for solar_term in SOLAR_TERMS_MONTH[month]:
            nums = SOLAR_TERMS_C_NUMS[solar_term]
            C = nums[0] if year < 2000 else nums[1]
            # 2000 年的小寒、大寒、立春、雨水按照 20 世纪的 C 值来算
            if year == 2000 and solar_term in [
                SolarTerms.lesser_cold,
                SolarTerms.greater_cold,
                SolarTerms.the_beginning_of_spring,
                SolarTerms.rain_water,
            ]:
                C = nums[0]
            Y = year % 100
            L = int(Y / 4)
            if solar_term in [
                SolarTerms.lesser_cold,
                SolarTerms.greater_cold,
                SolarTerms.the_beginning_of_spring,
                SolarTerms.rain_water,
            ]:
                L = int((Y - 1) / 4)
            day = int(Y * D + C) - L
            # 计算偏移量
            delta = SOLAR_TERMS_DELTA.get((year, solar_term))
            if delta:
                day += delta
            _date = datetime.date(year, month, day)
            if _date < start or _date > end:
                continue
            result.append((_date, solar_term.value[1]))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return result
    
    
def get_tgdz_year(year=None):
    '''计算（农历）年份天干地支'''
    if pd.isnull(year):
        year = datetime.datetime.now().year
    # 农历1984年是甲子年
    year = int(year)
    if year >= 1984:
        return TGDZ[(year-1984) % 60]
    else:
        return TGDZ[-((1984-year) % 60)]
    

def get_tgdz_date(date=None):
    '''根据公历日期计算农历干支纪日'''
    if pd.isnull(date):
        date = datetime.datetime.now().date()
    date = str(date)
    days = (pd.to_datetime(date) - TGDZ_BASE_DATE).days
    if days >= 0:
        return TGDZ[days % 60]
    else:
        return TGDZ[-(abs(days) % 60)]
    
    
def get_tgdz_hour(time=None):
    '''根据公历时间（小时）计算农历干支纪时'''
    if pd.isnull(time):
        time = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    time = str(time)
    dif = pd.to_datetime(time) - TGDZ_BASE_TIME
    days = dif.days
    seconds = dif.seconds + days*24*3600
    hours2 = seconds / 7200
    if hours2 >= 0:
        return TGDZ[int(hours2 % 60)]
    else:
        return TGDZ[-(int(abs(hours2) % 60))-1]

    
def get_bazi(time=None):
    '''
    根据公历时间生成八字
    
    Examples
    --------
    >>> get_bazi('1992.05.14 18:00')
    '''
    return gen2gz(time) + ',' + get_tgdz_hour(time) + '时'


def get_bazi_lunar(time, run=False):
    '''
    | 根据农历时间生成八字
    | time格式如'2023.02.30 19:30:20'，时分秒可以不写
    | run为True表示闰月日期
    
    Examples
    --------
    >>> get_bazi_lunar('2023.02.30')
    '''
    assert isinstance(time, str) and '.' in time
    date = lun2gen(time[:10], run=run)
    time = date + time[10:]
    return get_bazi(time)


def get_wuxing(time=None):
    bazi = get_bazi(time)
    wx_ = {x: tgdznywx[x[:2]]+'(%s, %s)'%(tgwx[x[0]], dzwx[x[1]]) for x in bazi.split(',')}
    wx = [v[2] for k, v in wx_.items()]
    return wx, wx_


def get_wuxing_lunar(time, run=False):
    bazi = get_bazi_lunar(time, run=run)
    wx_ = {x: tgdznywx[x[:2]]+'(%s, %s)'%(tgwx[x[0]], dzwx[x[1]]) for x in bazi.split(',')}
    wx = [v[2] for k, v in wx_.items()]
    return wx, wx_


def get_zodiac_match(time=None):
    '''根据公历时间获取属相合婚信息'''
    sx = gen2gz(time)[3]
    return {sx: zodiac_match[gen2gz(time)[3]]}


def get_zodiac_match_lunar(time, run=False):
    '''根据农历时间获取属相合婚信息'''
    assert isinstance(time, str) and '.' in time
    date = lun2gen(time[:10], run=run)
    sx = gen2gz(date)[3]
    return {sx: zodiac_match[gen2gz(time)[3]]}


def _hour2dz(hour):
    assert hour >= 0 and hour <= 23
    if hour >= 23 or hour < 1:
        return '子'
    elif hour >= 1 and hour < 3:
        return '丑'
    elif hour >= 3 and hour < 5:
        return '寅'
    elif hour >= 5 and hour < 7:
        return '卯'
    elif hour >= 7 and hour < 9:
        return '辰'
    elif hour >= 9 and hour < 11:
        return '巳'
    elif hour >= 11 and hour < 13:
        return '午'
    elif hour >= 13 and hour < 15:
        return '未'
    elif hour >= 15 and hour < 17:
        return '申'
    elif hour >= 17 and hour < 19:
        return '酉'
    elif hour >= 19 and hour < 21:
        return '戌'
    elif hour >= 21 and hour < 23:
        return '亥'


def _trans_hour(time=None):
    if pd.isnull(time):
        hour = datetime.datetime.now().hour
    else:
        hour = pd.to_datetime(str(time)).hour
    return _hour2dz(hour)


def fate_weight(time=None):
    '''称命，传入公历时间'''
    bazi = get_bazi(time)
    date = gen2lun(time)
    wy = float(w_year[bazi[:5]])
    wm = float(w_month[date[5:7]])
    wd = float(w_date[date[8:]])
    wh = float(w_hour[_trans_hour(time)])
    w = float(round(wy+wm+wd+wh, 2))
    sing = song[str(w)]
    result = {
        'weight': w,
        'bazi': bazi,
        'song': sing,
        'weight_split': (wy, wm, wd, wh)
    }
    return result


def fate_weight_lunar(time, run=False):
    '''
    称命，传入农历时间
    
    Examples
    --------
    >>> fate_weight_lunar('2023.02.30 09:30:00')
    '''
    assert isinstance(time, str) and '.' in time
    bazi = get_bazi_lunar(time, run=run)
    wy = float(w_year[bazi[:5]])
    wm = float(w_month[time[5:7]])
    wd = float(w_date[time[8:10]])
    hour = int(time[11:13]) if len(time) >= 13 else 0
    wh = float(w_hour[_hour2dz(hour)])
    w = float(round(wy+wm+wd+wh, 2))
    sing = song[str(w)]
    result = {
        'weight': w,
        'bazi': bazi,
        'song': sing,
        'weight_split': (wy, wm, wd, wh)
    }
    return result
    
    
def get_xingzuo(date=None):
    '''获取星座'''
    date = _wrap_date(date)
    md = date.strftime('%m-%d')
    if md >= '12-22' or md <= '01-19':
        return '摩羯座'
    elif '01-20' <= md <= '02-18':
        return '水瓶座'
    elif '02-19' <= md <= '03-20':
        return '双鱼座'
    elif '03-21' <= md <= '04-19':
        return '白羊座'
    elif '04-20' <= md <= '05-20':
        return '金牛座'
    elif '05-21' <= md <= '06-21':
        return '双子座'
    elif '06-22' <= md <= '07-22':
        return '巨蟹座'
    elif '07-23' <= md <= '08-22':
        return '狮子座'
    elif '08-23' <= md <= '09-22':
        return '处女座'
    elif '09-23' <= md <= '10-23':
        return '天秤座'
    elif '10-23' <= md <= '11-22':
        return '天蝎座'
    elif '11-23' <= md <= '12-21':
        return '射手座'


def get_recent_workday(date=None, dirt='post'):
    '''
    若date为工作日，则返回，否则返回下一个(post)或上一个(pre)工作日
    '''
    date = _trans_date(date)
    tdelta = datetime.timedelta(1)
    if dirt == 'post':
        while not is_workday(date):
            date =  date + tdelta
    elif dirt == 'pre':
        while not is_workday(date):
            date =  date - tdelta
    return _wrap_date(date)


def get_next_nth_workday(date=None, n=1):
    '''
    | 给定日期date，返回其后第n个工作日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    '''
    date = _trans_date(date)
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = date = date + datetime.timedelta(n_add)
        if is_workday(date):
            tmp += 1
    return _wrap_date(date)


def get_work_dates(start_date, end_date=None):
    '''
    取指定起止日期内的工作日
    '''
    start_date = _trans_date(start_date)
    end_date = _trans_date(end_date)
    dates = get_workdays(start_date, end_date)
    dates = [_wrap_date(x) for x in dates]
    return dates


def _is_tradeday(date):
    return is_workday(date) and date.weekday() not in [5, 6]


def is_tradeday(date=None, market='SSE'):
    '''判断是否为交易日'''
    market = market.upper()
    if not market in MARKETS:
        raise ValueError('未识别的交易所，请检查！')
    date = _wrap_date(date)    
    if date < MARKETS[market]: # 小于首个交易日的直接视为非交易日
        return False
    if (market, date) in trade_dates:
        return bool(trade_dates[(market, date)])
    return _is_tradeday(date)


def get_recent_tradeday(date=None, dirt='post', market='SSE'):
    '''
    若date为交易日，则直接返回date，否则返回下一个(dirt='post')或上一个(dirt='pre')交易日
    '''
    assert dirt in ['post', 'pre']
    date = _trans_date(date)
    tdelta = datetime.timedelta(1)
    if dirt == 'post':
        while not is_tradeday(date, market=market):
            date = date + tdelta
    elif dirt == 'pre':
        while not is_tradeday(date, market=market):
            date = date - tdelta
    return _wrap_date(date)


def get_next_nth_tradeday(date=None, n=1, market='SSE'):
    '''
    | 给定日期date，返回其后第n个交易日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    '''
    date = _trans_date(date)
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = date + datetime.timedelta(n_add)
        if is_tradeday(date, market=market):
            tmp += 1
    return _wrap_date(date)


def get_trade_dates(start_date, end_date=None, market='SSE'):
    '''
    取指定起止日期内的交易日期（周内的工作日）
    '''
    start_date = _trans_date(start_date)
    end_date = _trans_date(end_date)
    dates = pd.date_range(start_date, end_date)
    dates = [x for x in dates if is_tradeday(x, market=market)]
    dates = [_wrap_date(x) for x in dates]
    return dates


def _to_dot(date):
    if pd.isnull(date):
        date = datetime.datetime.now()
    date = pd.to_datetime(str(date)).strftime('%Y.%m.%d')
    return date


def gen2lun(date=None):
    '''公历日期转农历日期'''
    return gen_lun[_to_dot(date)]


def gen2gz(date=None):
    '''公历日期转干支纪日法'''
    return gen_gz[_to_dot(date)]


def lun2gen(date, run=False):
    '''
    | 农历日期转普通日期
    | date格式如'2023.02.30'
    | run为True表示闰月日期
    
    Examples
    --------
    >>> lun2gen('2023.02.30')
    '''
    assert isinstance(date, str) and len(date) == 10 and '.' in date
    if run:
        date = date + '闰'
    if date in lun_gen:
        return lun_gen[date]
    else:
        raise ValueError('未找到对应农历日期，请检查输入参数！')
        
        
def get_age_by_shuxiang(shuxiang, return_n=10):
    '''根据属性获取可能年龄'''
    assert isinstance(shuxiang, str) and shuxiang in SX
    base_year, base_sx = 2022, '虎'
    res = {}
    n = 0
    year, idx = base_year, SX.index(base_sx)
    while n < return_n:
        if SX[idx] == shuxiang:
            res[year] = base_year-year
            n += 1
        year -= 1
        if idx == 0:
            idx = 11
        else:
            idx -= 1    
    return res
    
