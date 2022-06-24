# -*- coding: utf-8 -*-

import re
import time
import datetime
import pandas as pd
from chinese_calendar import is_workday


def str2datetime(tstr, strformat='%Y-%m-%d'):
    '''时间字符串转datetime格式'''
    return datetime.datetime.strptime(tstr, strformat)


def str2timestamp(t, strformat='%Y-%m-%d %H:%M:%S'):
    '''
    时间字符串转时间戳(精确到秒)
    '''
    return time.mktime(time.strptime(t, strformat))


def timestamp2str(t, strformat=None):
    '''    
    时间戳转化为字符串

    Parameters
    ----------
    t : str, int, float
        | 时间戳，可为数字字符串、整数或浮点数
        | 当t取整后位数小于等于10位时，精度按秒处理
        | 当t取整后位数大于10位小于等于13位时，精度按毫秒处理
        | 当t取整后位数大于13位时，精度按微秒处理
    strformat : str, None
        返回时间格式，默认 ``%Y-%m-%d %H:%M:%S``

    
    :returns: `str` - 字符串格式时间


    .. hint::
      | 正常到秒级别的时间戳为10位整数
      | 到毫秒/千分之一秒(milliseconds)级别的时间戳为13位整数
      | 到微秒/百万分之一秒(microseconds)级别的时间戳为16位整数
    
    TODO
    ----
    纳秒级别精度添加
    '''

    tlen = len(str(int(t)))
    l = 10 ** (max(tlen-10, 0))
    ts, ms = divmod(float(t), l)
    dt = datetime.datetime.fromtimestamp(ts)

    if tlen > 13:
        dt = dt + datetime.timedelta(microseconds=ms)
    elif tlen > 10:
        dt = dt + datetime.timedelta(milliseconds=ms)

    if strformat is None:
        # 到微秒
        # dt = dt.isoformat(sep=' ', timespec='microseconds')
        # dt = dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        # 到毫秒
        # dt = dt.isoformat(sep=' ', timespec='milliseconds')
        # dt = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # 到秒
        # dt = dt.isoformat(sep=' ', timespec='seconds')
        dt = dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        dt = dt.strftime(strformat)

    return dt


def datetime_now(strformat=None):
    '''
    获取当前时间日期
    '''
    t = time.time()
    return timestamp2str(t, strformat=strformat)


def get_date_format(date, joiners=[' ', '-', '/', '*', '#', '@', '.', '_']):
    '''
    判断日期格式位，返回日期位数和连接符

    Parameters
    ----------
    date : str
        表示日期的字符串
    joiners : list
        支持的格式连接符，不支持的连接符可能报错或返回'未知日期格式'

        .. note:: 注：若date无格式连接符，则只判断8位的格式，如'20200220'


    :returns: tuple - (日期位数（可能为date字符串长度或'未知日期格式'）, 格式连接符（或None）)
    '''

    # if isinstance(date, int):
    #     tmp = str(date)
    #     if len(tmp) == 8:
    #         return 8, 'int'
    #     else:
    #         return '未知日期格式', None
    assert isinstance(date, str), '只支持string格式日期'

    for joiner in joiners:
        reg = re.compile(r'\d{4}['+joiner+']\d{2}['+joiner+']\d{2}')
        if len(reg.findall(date)) > 0:
            return len(date), joiner

    # 这里只要8位都是数字就认为date是形如'20200226'格式的日期
    if len(date) == 8:
        tmp = [str(x) for x in range(0, 10)]
        tmp = [x in tmp for x in date]
        if all(tmp):
            return 8, ''

    return '未知日期格式', None


def get_year_month(date):
    '''
    | 提取date中的年月
    | 例：'20200305'—>'202003', '2020-03-05'—>'2020-03'
    '''
    n, joiner = get_date_format(date)
    if n == 8 and joiner == '':
        return date[0:-2]
    if joiner is not None:
        return joiner.join(date.split(joiner)[0:2])
    raise ValueError('请检查日期格式：只接受get_date_format函数识别的日期格式！')


def today_date(joiner='-', forbid_joiners=['%']):
    '''
    获取今日日期，格式由连接符joiner确定，forbid_joiners指定禁用的连接符
    '''
    if joiner in forbid_joiners:
        raise ValueError('非法连接符：{}！'.format(joiner))
    return datetime.datetime.now().strftime(joiner.join(['%Y', '%m', '%d']))


def today_month(joiner='-', forbid_joiners=['%']):
    '''获取今日所在年月'''
    return get_year_month(today_date(joiner, forbid_joiners))


def today_quarter(joiner='Q'):
    '''获取今日所在季度'''
    t = datetime.datetime.now()
    year, month = t.year, t.month
    def m_q(m):
        assert m < 13
        if m < 4:
            return 1
        elif m < 7:
            return 2
        elif m < 10:
            return 3
        elif m < 13:
            return 4
    return joiner.join([str(year), str(m_q(month))])


def date_reformat_simple(date, joiner_ori, joiner_new='-',
                         forbid_joiners=['%']):
    '''时间格式简单转化'''
    if joiner_new in forbid_joiners:
        raise ValueError('非法连接符：{}！'.format(joiner_new))
    return joiner_new.join([x.zfill(2) for x in date.split(joiner_ori)])


def date_reformat_chn(date, joiner='-', forbid_joiners=['%']):
    '''
    | 指定连接符为joiner，重新格式化date，date格式为x年x月x日.
    | forbid_joiners指定禁用的连接符
    '''
    return date_reformat_simple(
        date.replace('年', '-').replace('月', '-').replace('日', ''),
        joiner_ori='-', joiner_new=joiner, forbid_joiners=forbid_joiners)


def date_reformat(date, joiner='-', forbid_joiners=['%']):
    '''指定连接符为joiner，重新格式化date，forbid_joiners指定禁用的连接符'''
    if joiner in forbid_joiners:
        raise ValueError('非法连接符：{}！'.format(joiner))
    n, joiner_ori = get_date_format(date)
    if joiner_ori is not None:
        formater_ori = joiner_ori.join(['%Y', '%m', '%d'])
        date = datetime.datetime.strptime(date, formater_ori)
        formater = joiner.join(['%Y', '%m', '%d'])
        return date.strftime(formater)
    raise ValueError('请检查日期格式：只接受get_date_format函数识别的日期格式！')


def date8_to_date10(date, joiner='-'):
    '''8位日期转换为10位日期，连接符为joiner'''
    n, _ = get_date_format(date)
    if n != 8:
        raise ValueError('请检查日期格式，接受8位且get_date_format函数识别的日期格式！')
    return joiner.join([date[0:4], date[4:6], date[6:]])


def date10_to_date8(date):
    '''10位日期转换为8位日期'''
    n, joiner = get_date_format(date)
    if n != 10 or joiner is None:
        raise ValueError('请检查日期格式，接受10位且get_date_format函数识别的日期格式！')
    return ''.join(date.split(joiner))


def date_add_nday(date, n=1):
    '''
    | 在给定日期date上加上n天（减去时n写成负数即可）
    | 日期输入输出符合get_date_format函数支持的格式
    '''
    _, joiner = get_date_format(date)
    if joiner is not None:
        formater = joiner.join(['%Y', '%m', '%d'])
        date = datetime.datetime.strptime(date, formater)
        date_delta = datetime.timedelta(days=n)
        date_new = date + date_delta
        return date_new.strftime(formater)
    raise ValueError('请检查日期格式：只接受get_date_format函数识别的日期格式！')


def diff_time_second(time1, time2):
    '''
    | 计算两个时间的差：time1-time2，time1和time2的格式应相同
    | 返回两个时间差的秒数
    '''
    time1_ = pd.to_datetime(time1)
    time2_ = pd.to_datetime(time2)
    time_delta = time1_ - time2_
    return 24*3600*time_delta.days + time_delta.seconds


def diff_days_date(date1, date2):
    '''
    | 计算两个日期间相隔天数，若date1大于date2，则输出为正，否则为负
    | 日期输入输出符合get_date_format函数支持的格式
    '''
    n1, joiner1 = get_date_format(date1)
    n2, joiner2 = get_date_format(date2)
    if joiner1 is not None and joiner2 is not None:
        formater1 = joiner1.join(['%Y', '%m', '%d'])
        formater2 = joiner2.join(['%Y', '%m', '%d'])
        date1 = datetime.datetime.strptime(date1, formater1)
        date2 = datetime.datetime.strptime(date2, formater2)
        return (date1-date2).days
    raise ValueError('请检查日期格式：只接受get_date_format函数识别的日期格式！')


def get_dates_between(date1, date2, keep1=False, keep2=True,
                      only_workday=False, del_weekend=False,
                      joiner=2):
    '''
    获取date1到date2之间的日期列表，keep1和keep2设置结果是否保留date1和date2


    .. note:: 是否为workday用了chinese_calendar包，若chinese_calendar库没更新，可能导致结果不准确
    '''
    _, joiner1 = get_date_format(date1)
    _, joiner2 = get_date_format(date2)
    date1 = date_reformat(date1, '-')
    date2 = date_reformat(date2, '-')
    dates = pd.date_range(date1, date2)
    if only_workday:
        dates = [x for x in dates if is_workday(x)]
    if del_weekend:
        dates = [x for x in dates if x.weekday() not in [5, 6]]
    dates = [x.strftime('%Y-%m-%d') for x in dates]
    if not keep1:
        dates = dates[1:]
    if not keep2:
        dates = dates[:-1]
    if joiner == 2:
        joiner = joiner2
    elif joiner == 1:
        joiner = joiner1
    dates = [date_reformat(x, joiner) for x in dates]
    return dates


def get_dayofweek(date=None):
    '''返回date属于星期几（1-7）'''
    if pd.isnull(date):
        date = today_date()
    return pd.to_datetime(date_reformat(date, '-')).weekday() + 1


def get_dayofyear(date=None):
    '''返回date属于一年当中的第几天（从1开始记）'''
    if pd.isnull(date):
        date = today_date()
    return pd.to_datetime(date_reformat(date, '-')).dayofyear


def isworkday_chncal(date=None):
    '''利用chinese_calendar库判断date（str格式）是否为工作日'''
    if pd.isnull(date):
        date = today_date()
    date = date_reformat(date, '')
    date_dt = datetime.datetime.strptime(date, '%Y%m%d')
    return is_workday(date_dt)


def get_recent_workday_chncal(date=None, dirt='post'):
    '''
    若date为工作日，则返回，否则返回下一个(post)或上一个(pre)工作日


    .. note:: 若chinese_calendar库没更新，可能导致结果不准确
    '''
    if pd.isnull(date):
        date = today_date()
    if dirt == 'post':
        while not isworkday_chncal(date):
            date = date_add_nday(date, 1)
    elif dirt == 'pre':
        while not isworkday_chncal(date):
            date = date_add_nday(date, -1)
    return date


def get_next_nth_workday_chncal(date=None, n=1):
    '''
    | 给定日期date，返回其后第n个工作日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    
    
    .. note:: 若chinese_calendar库没更新，可能导致结果不准确
    '''
    if pd.isnull(date):
        date = today_date()
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = date_add_nday(date, n_add)
        if isworkday_chncal(date):
            tmp += 1
    return date


def get_recent_inweekday(date=None, dirt='post'):
    '''
    若date不是周末，则返回，否则返回下一个(post)或上一个(pre)周内日期
    '''
    if pd.isnull(date):
        date = today_date()
    if dirt == 'post':
        while not get_dayofweek(date) <= 5:
            date = date_add_nday(date, 1)
    elif dirt == 'pre':
        while not get_dayofweek(date) <= 5:
            date = date_add_nday(date, -1)
    return date


def get_next_nth_inweekday(date=None, n=1):
    '''
    | 给定日期date，返回其后第n个周内日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    '''
    if pd.isnull(date):
        date = today_date()
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = date_add_nday(date, n_add)
        if get_dayofweek(date) <= 5:
            tmp += 1
    return date


def cut_date(start_date, end_date, n=500):
    '''
    以间隔为n天划分start_date和end_date之间日期子集

    Examples
    --------
    >>> start_date, end_date, n = '20200201', '20200225', 10
    >>> cut_date(start_date, end_date, n)
    [['20200201', '20200210'],
     ['20200211', '20200220'],
     ['20200221', '20200225']]
    '''
    dates = []
    tmp = start_date
    while tmp <= end_date:
        dates.append([tmp, date_add_nday(tmp, n-1)])
        tmp = date_add_nday(tmp, n)
    if dates[-1][-1] > end_date:
        dates[-1][-1] = end_date
    return dates


def diff_workdays_date(date1, date2):
    '''
    | 计算两个日期间相隔的工作日天数，若date1大于date2，则输出为正，否则为负
    | 日期输入输出符合get_date_format函数支持的格式
    '''
    raise NotImplementedError
