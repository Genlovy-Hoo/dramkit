# -*- coding: utf-8 -*-

import re
import time
import datetime
import pandas as pd

from dramkit import chncal


# Python关于时间的一些知识：
# 世界标准时间有两个：
# GMT时间(格林威治时间)：以前的世界时，本地时间根据时区差异基于GMT进行调整，比如北京时间是东八区，比GMT时间早8小时
# （GMT是根据地球自转和公转来计时的，地球自转一圈是一天，绕太阳公转一圈是一年，GMT缺点是地球自转一圈的时间并不是恒定的）
# UTC时间：现在世界标准时/世界协调时间，本地时间根据偏移量基于UTC进行调整，比如北京时间偏移量是+0800，即比UTC早8个小时
# （UTC时间认为一个太阳日（一天）总是恒定的86400秒（24小时））
# 时间戳：以某个时间点为基准，按秒为单位计数，通过整数或浮点数来记录时间的方式
# UNIX时间戳：以UTC时间1970-01-01 00:00:00为基准的时间戳
# 本地时间戳：以某个本地时间为基准的时间戳（我们用的北京时间戳通常以北京时间1970-01-01 08:00:00为基准）
# 对同一个字符时间，转为时间戳时，UTC时间戳比北京时间戳大8*3600
# 对同一个时间戳，转为为字符时间，UTC时间比北京时间小8小时
# time和datetime模块默认时间戳基准是本地（北京）时间1970-01-01 08:00:00
# pandas模块默认时间戳基准是UTC时间1970-01-01 00:00:00
# 参考：
# https://mp.weixin.qq.com/s/VdoQt88JfjPJTL9XgohZJQ
# https://zhuanlan.zhihu.com/p/412552917
# http://t.zoukankan.com/Cheryol-p-13479418.html
# https://www.zhihu.com/question/400394490/answer/1273564089


# pandas模块和datetime以及time模块的时间戳有差别（pd多8小时(28800秒)，后两者是一致的）
TSDIF_PD_DTTM = pd.to_datetime('19700102 08:00:00').timestamp() - \
                datetime.datetime(1970, 1, 2, 8, 0, 0).timestamp()
TS_BIAS_DT = (datetime.datetime.fromtimestamp(0) - \
              datetime.datetime.utcfromtimestamp(0)).seconds
    
MONTH_DAYS = {1: 31, 2: 28, 3:31, 4: 30, 5:31, 6:30, 7: 31,
              8: 31, 9: 30, 10: 31, 11:30, 12: 31}


def _show_time_infos():
    t, t_ns = time.time(), time.time_ns()
    print('原始时间戳:                ', t)
    print('秒级(second, 10位):       ', int(t))
    print('毫秒级(millisecond, 13位):', int(round(t*1000)))
    print('微秒级(microsecond, 16位):', int(round(t*1000000)))
    print('纳秒级(nanosecond, 19位) :', int(round(t*1000000000)))
    print('纳秒级(nanosecond, 19位) :', t_ns)
    
    print('time模块时区名称:', time.tzname)
    print('time模块当前时区时间差:', time.timezone)
    
    fmt = '%Y-%m-%d %H:%M:%S'
    print('time.time()时间戳对应当地时间:',
          time.strftime(fmt, time.localtime(t)))
    print('time.time()时间戳对应UTC时间: ',
          time.strftime(fmt, time.gmtime(t)))
    
    now = time.strftime(fmt)
    print('当前时间pandas datetime时间戳:',
          pd.to_datetime(now).timestamp())
    print('当前时间datetime模块时间戳:    ',
          datetime.datetime.strptime(now, fmt).timestamp())
    print('当前时间time模块时间戳:        ',
          time.mktime(time.strptime(now, fmt)))
    
    print('pandas datetime时间戳起始时间:', pd.to_datetime(0))
    print('datetime模块时间戳起始时间:    ', datetime.datetime.fromtimestamp(0))
    print('time模块时间戳起始时间(当地):   ', time.strftime(fmt, time.localtime(0)))
    print('time模块时间戳起始时间(UTC):   ', time.strftime(fmt, time.gmtime(0)))


def pd_str2datetime(o, **kwargs):
    '''pd.to_datetime函数封装，排除o为时间戳（统一按str处理）'''
    if type(o).__name__ in ['Series', 'ndarray']:
        try:
            return pd.to_datetime(o.astype(int).astype(str), **kwargs)
        except:
            try:
                return pd.to_datetime(o.astype(str), **kwargs)
            except:
                return pd.to_datetime(o, **kwargs)
    else:
        try:
            return pd.to_datetime(str(o), **kwargs)
        except:
            try:
                return pd.to_datetime(str(int(o)), **kwargs)
            except:
                return pd.to_datetime(o, **kwargs)
            
            
def datetime2str(dt, strformat=None, tz='local'):
    '''
    datatime格式转为str格式
    
    Parameters
    ----------
    dt : datetime.datetime, datetime.date, time.struct_time, pd.Timestamp
        datetime、time或pandas格式时间
    strformat : None, bool, str
        设置返回文本格式：
        
        - False, 直接返回dt
        - 'timestamp', 返回timestamp
        - 'int', 返回int格式日期，如'20221106'
        - None, 默认'%Y-%m-%d %H:%M:%S'格式        
        - 指定其他文本格式
    tz : str
        指定时区，可选['local', 'utc']
        
    Examples
    --------
    >>> dt1, dt2 = datetime.datetime.now(), time.localtime()
    >>> dt3 = pd_str2datetime(datetime_now())
    >>> datetime2str(dt1, 'timestamp', 'utc')
    >>> datetime2str(dt2, 'timestamp', 'utc')
    >>> datetime2str(dt3, 'timestamp', 'utc')
    >>> dt4 = time.gmtime()
    >>> datetime2str(dt3, 'timestamp', 'utc')
    >>> datetime2str(dt3, 'timestamp')
    '''
    if strformat == False:
        return dt
    if strformat is None:
        # strformat = '%Y-%m-%d %H:%M:%S.%f' # 到微秒
        strformat = '%Y-%m-%d %H:%M:%S' # 到秒
        # strformat = '%Y-%m-%d' # 到日
    dt_type = type(dt).__name__
    assert dt_type in ['datetime', 'date', 'Timestamp', 'struct_time']
    assert tz in ['local', 'utc']
    if strformat == 'int':
        try:
            return int(dt.strftime('%Y%m%d'))
        except:
            return int(time.strftime('%Y%m%d', dt))
    elif strformat == 'timestamp':
        if dt_type in ['datetime', 'date']:
            if tz == 'local':
                return dt.timestamp()
            else:
                return dt.timestamp() + TS_BIAS_DT
        elif dt_type == 'struct_time':
            if tz == 'local':
                return time.mktime(dt)
            else:
                return time.mktime(dt) - time.timezone
        else:
            if tz == 'local':
                return dt.timestamp() - TSDIF_PD_DTTM
            else:
                return dt.timestamp()
    else:
        try:
            try:
                res = dt.strftime(strformat)
            except:
                res = time.strftime(strformat, dt)
            if res == strformat:
                raise
            if not pd.isna(pd_str2datetime(res, format=strformat)):
                return res
            raise
        except:
            if dt_type == 'struct_time':
                return time.strftime('%Y-%m-%d %H:%M:%S.%f', dt)
            else:
                return str(dt)
            
            
def dtseries2str(series, joiner='-', strformat=None):
    '''pd.Series转化为str，若不指定strformat，则按joiner连接年月日'''
    res = pd.to_datetime(series)
    if pd.isnull(strformat):
        strformat = '%Y{x}%m{x}%d'.format(x=joiner)
    res = res.apply(lambda x: x.strftime(strformat))
    return res


def get_datetime_strformat(tstr,
                           ymd = ['-', '', '.', '/'],
                           ymd_hms = [' ', '', '.'],
                           hms = [':', ''],
                           hms_ms = ['.', '']
                           ):
    '''
    | 获取常用的文本时间格式format，支持包含以下几个部分的时间类型：
    | 年月日、年月日时分秒、年月日时分、年月日时分秒毫秒、年月日时、年月
    
    TODO
    ----
    改为用正则表达式匹配（根据连接符、数字占位限制和数字大小限制来匹配）
    
    Parameters
    ----------
    ymd : list
        年月日连接符
    ymd_hms : list
        年月日-时分秒连接符
    hms : list
        时分秒连接符
    hms_ms : list
        时分秒-毫秒连接符
    '''
    assert isinstance(tstr, str), '只接受文本格式时间！'
    for j1 in ymd:
        try:
            # 年月日
            fmt1 = '%Y{j1}%m{j1}%d'.format(j1=j1)
            _ = time.strptime(tstr, fmt1)
            if len(tstr) == 8+2*len(j1):
                return fmt1
            else:
                raise
        except:
            for j2 in ymd_hms:
                for j3 in hms:
                    try:
                        # 年月日时分秒
                        fmt2 = fmt1 + '{j2}%H{j3}%M{j3}%S'.format(j2=j2, j3=j3)
                        _ = time.strptime(tstr, fmt2)
                        if len(tstr) == 14+2*len(j1)+len(j2)+2*len(j3):
                            return fmt2
                        else:
                            raise
                    except:
                        try:
                            # 年月日时分
                            fmt21 = fmt1 + '{j2}%H{j3}%M'.format(j2=j2, j3=j3)
                            _ = time.strptime(tstr, fmt21)
                            if len(tstr) == 12+2*len(j1)+len(j2)+len(j3):
                                return fmt21
                            else:
                                raise
                        except:
                            for j4 in hms_ms:
                                try:
                                    # 年月日时分秒毫秒
                                    fmt3 = fmt2 + '{j4}%f'.format(j4=j4)
                                    _ = time.strptime(tstr, fmt3)
                                    if len(tstr) > 14+2*len(j1)+len(j2)+2*len(j3)+len(j4):
                                        len1 = 14+2*len(j1)+len(j2)+2*len(j3)
                                        tstr1 = tstr[:len1]
                                        _ = time.strptime(tstr1, fmt2)
                                        if len(tstr1) == len1:
                                           return fmt3
                                        else:
                                            raise
                                    else:
                                        raise
                                except:
                                    try:
                                        # 年月日时
                                        fmt22 = fmt1 + '{j2}%H'.format(j2=j2)
                                        _ = time.strptime(tstr, fmt22)
                                        if len(tstr) == 10+2*len(j1)+len(j2):
                                            return fmt22
                                        else:
                                            raise
                                    except:
                                        try:
                                            # 年月
                                            fmt11 = '%Y{j1}%m'.format(j1=j1)
                                            _ = time.strptime(tstr, fmt11)
                                            if len(tstr) == 6+len(j1):
                                                return fmt11
                                            else:
                                                raise
                                        except:
                                            pass
    raise ValueError('未识别的日期时间格式！')
    
    
def str2datetime(tstr, strformat=None):
    '''时间字符串转datetime格式'''
    return pd.to_datetime(tstr, format=strformat)
    # if strformat is None:
    #     strformat = get_datetime_strformat(tstr)
    # return pd.to_datetime(tstr, format=strformat)
    
    
def str2timestamp(t, strformat=None, tz='local'):
    '''
    | 时间字符串转时间戳
    | 若大于16位(纳秒级)，返回string，否则返回float(毫秒级)
    '''
    assert tz in ['local', 'utc']
    t = str2datetime(t, strformat=strformat)
    ts = t.timestamp()
    if tz == 'local':
        ts = ts - TSDIF_PD_DTTM
    if t.nanosecond > 0:
        ts = str(ts) + str(t.nanosecond)
    return ts
    
    
def timestamp2str(t, strformat=None, tz='local', method=2):
    '''    
    | 时间戳转化为字符串格式
    | 注意：t小于0时直接按t为秒数处理（即使整数部分超过10位也不会处理为小数）
    '''
    assert tz in ['local', 'utc']
    assert method in [1, 2, 3] # 经测试method=2速度最快
    assert isinstance(t, (int, float, str)), '请检查时间戳格式！'
    assert isinstance(strformat, (type(None), str)), '请指定正确的输出格式！'
    strformat = '%Y-%m-%d %H:%M:%S' if strformat is None else strformat
    def _delta2str(seconds):
        if tz == 'local':
            tbase = datetime.datetime.fromtimestamp(0)
        else:
            tbase = datetime.datetime.utcfromtimestamp(0)
        dt = tbase + datetime.timedelta(seconds=seconds)
        return dt.strftime(strformat)        
    # t小于0特殊处理
    if float(t) < 0:
        return _delta2str(float(t))
    # 先转化为时间戳数字（整数部分大于10位的转化为小数）
    ts = str(t).replace('.', '')
    if len(ts) > 10:
        part_int = ts[:10]
        part_float = ts[10:]
        ts = int(part_int) + int(part_float) / (10**len(part_float))
    else:
        ts = float(t)
    # 方式一：用datetime.fromtimestamp
    if method == 1:
        if tz == 'local':
            dt = datetime.datetime.fromtimestamp(ts)
        else:
            dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.strftime(strformat)
    # 方式二：用time.localtime
    if method == 2:
        if tz == 'local':
            dt = time.localtime(ts)
        else:
            dt = time.gmtime(ts)
        return time.strftime(strformat, dt)
    # 方式三：用timedelta转化
    if method == 3:
        return _delta2str(ts)

    
def get_datetime_format(dt,
                        ymd = ['-', '', '.', '/'],
                        ymd_hms = [' ', '', '.'],
                        hms = [':', ''],
                        hms_ms = ['.', '']
                        ):
    '''
    | 获取常用日期时间格式format
    | 注：整数和浮点数都可能是时间戳，这里以常用的规则来判断：
    |    - 若整数长度为8位，判断为整数型日期
    |    - 若整数或浮点数长度不小于10位不大于19位，判断为时间戳
    |    - 其他情况下的整数或浮点数报错处理
    '''
    dt_type = type(dt).__name__
    if dt_type in ['datetime', 'date']:
        return 'datetime.' + dt_type, None
    if dt_type == 'Timestamp':
        return 'pd.' + dt_type, None
    if dt_type == 'struct_time':
        return 'time.' + dt_type, None
    if isinstance(dt, (int, float, str)):
        try:
            fmt = get_datetime_strformat(str(dt), ymd=ymd,
                                         ymd_hms=ymd_hms,
                                         hms=hms, hms_ms=hms_ms)
            return type(dt).__name__, fmt
        except:
            try:
                _ = float(dt)
                if 10 <= len(str(dt)) <= 19:
                    return type(dt).__name__, 'timestamp'
            except:
                pass
    raise ValueError('未识别的日期时间格式！')

    
def copy_format0(to_tm, from_tm):
    '''
    | 复制日期时间格式
    | 若from_tm是日期时间格式或时间戳格式，则直接返回to_tm
    '''
    dtype, fmt = get_datetime_format(from_tm)
    if fmt in [None, 'timestamp']:
        return to_tm
    input_type = type(to_tm).__name__
    types1 = ['datetime', 'date', 'Timestamp', 'struct_time']
    types2 = ['ndarray', 'Series', 'list', 'tuple']
    assert input_type in types1+types2
    onlyone = False
    if not input_type in types2:
        onlyone = True
        to_tm = [to_tm]
    if len(to_tm) == 0:
        return to_tm                        
    dt_type = type(to_tm[0]).__name__
    assert dt_type in ['datetime', 'date', 'Timestamp', 'struct_time']
    if dt_type == 'struct_time':
        res = [time.strftime(fmt, x) for x in to_tm]
    else:
        res = [x.strftime(fmt) for x in to_tm]
    if dtype in ['int', 'float']:
        res = [eval('%s(x)'%dtype) for x in res]
    if onlyone:
        res = res[0]
    return res


def x2datetime(x, tz='local'):
    '''
    | x转化为datetime格式，若x为timestamp，应设置时区tz
    | 若x为8位整数，则转化为str处理，其余情况直接用用pd.to_datetime处理
    '''
    if isinstance(x, time.struct_time):
        return pd.to_datetime(
               datetime.datetime.fromtimestamp(time.mktime(x)))
    elif isinstance(x, int) and len(str(x)) == 8:
        return pd.to_datetime(str(x))
    else:
        try:
            xtype, fmt = get_datetime_format(x)
            if fmt == 'timestamp':
                return pd.to_datetime(timestamp2str(x, tz=tz))
            else:
                return pd.to_datetime(x)
        except:
            return pd.to_datetime(x)


def copy_format(to_tm, from_tm):
    '''
    复制日期时间格式
    '''
    if pd.isna(from_tm):
        return to_tm
    input_type = type(to_tm).__name__
    types1 = ['datetime', 'date', 'Timestamp', 'struct_time']
    types2 = ['str', 'int', 'float']
    types3 = ['ndarray', 'Series', 'list', 'tuple']
    assert input_type in types1+types2+types3
    if input_type == type(from_tm).__name__ and input_type in types1:
        return to_tm
    onlyone = False
    if not input_type in types3:
        onlyone = True
        to_tm = [to_tm]
    if len(to_tm) == 0:
        return to_tm
    def _return(res):
        if onlyone:
            return res[0]
        return res
    tz = 'local'
    res = [x2datetime(x, tz=tz) for x in to_tm]
    dtype, fmt = get_datetime_format(from_tm)
    if fmt is None:
        if dtype == 'time.struct_time':
            res = [x.timetuple() for x in res]
        return _return(res)
    if fmt == 'timestamp':
        # TODO: 不同格式的timestamp处理（如10位，13位，整数，小数等）
        res = [datetime2str(x, strformat='timestamp', tz=tz) for x in res]
        return _return(res)
    res = [x.strftime(fmt) for x in res]
    if dtype in ['int', 'float']:
        res = [eval('%s(x)'%dtype) for x in res]
    return _return(res)


def time_add(t, seconds=0, minutes=0, hours=0, days=0):
    '''时间运算，return_format指定返回格式，若为False，返回datetime格式'''
    seconds = seconds + 60*minutes + 60*60*hours + 60*60*24*days
    tdelta = datetime.timedelta(seconds=seconds)
    tnew = x2datetime(t) + tdelta
    return copy_format(tnew, t)


def date_add_nday(date, n=1):
    '''
    在给定日期date上加上n天（减去时n写成负数即可）
    '''
    return time_add(date, days=n)
    # date_delta = datetime.timedelta(days=n)
    # date_new = x2datetime(date) + date_delta
    # return copy_format(date_new, date)


def is_datetime(x):
    '''判断x是否为时间日期'''
    try:
        _ = pd.to_datetime(x)
        return True
    except:
        return False
    
    
def is_month_end(date=None, tz='local'):
    '''判断date是否为月末，若date为timestamp，应设置时区tz'''
    if date is None:
        date = datetime.date.today()
    date = x2datetime(date, tz=tz)
    return date.is_month_end


def is_month_start(date=None, tz='local'):
    '''判断date是否为月初，若date为timestamp，应设置时区tz'''
    if date is None:
        date = datetime.date.today()
    date = x2datetime(date, tz=tz)
    return date.is_month_start
    
    
def is_quarter_end(date=None, tz='local'):
    '''判断date是否为季末，若date为timestamp，应设置时区tz'''
    if pd.isnull(date):
        date = datetime.date.today()
    date = x2datetime(date, tz=tz)
    return date.is_quarter_end


def is_quarter_start(date=None, tz='local'):
    '''判断date是否为季末，若date为timestamp，应设置时区tz'''
    if pd.isnull(date):
        date = datetime.date.today()
    date = x2datetime(date, tz=tz)
    return date.is_quarter_start


def datetime_now(strformat=None):
    '''
    获取当前时间
    '''
    assert isinstance(strformat, (type(None), str, bool))
    if strformat == 'timestamp':
        return time.time()
    dt = datetime.datetime.now()
    return datetime2str(dt, strformat=strformat)
    # return dt.strftime('%Y-%m-%d %H:%M:%S' if strformat is None else strformat)
    # 到微秒
    # return dt.isoformat(sep=' ', timespec='microseconds')
    # return dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    # 到毫秒
    # return dt.isoformat(sep=' ', timespec='milliseconds')
    # return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def today_month(joiner='-'):
    '''获取今日所在年月，格式由连接符joiner确定'''
    if joiner.lower() == 'int':
        return int(datetime.date.today().strftime('%Y%m'))
    return datetime.date.today().strftime('%Y{}%m'.format(joiner))


def today_date(joiner='-'):
    '''获取今日日期，格式由连接符joiner确定'''
    if joiner.lower() == 'int':
        return int(datetime.date.today().strftime('%Y%m%d'))
    if joiner.lower() in ['dt', 'datetime']:
        return pd.to_datetime(datetime.date.today())
    return datetime.date.today().strftime(joiner.join(['%Y', '%m', '%d']))


def get_quarter(dt=None, joiner='Q'):
    '''获取日期所在季度'''
    if pd.isnull(dt):
        dt = datetime.datetime.now()
    t = x2datetime(dt)
    y, q = t.year, t.quarter
    if not joiner:
        return (y, q)
    return joiner.join([str(y), str(q)])


def get_quarter_start_end_by_yq(y, q, joiner='-'):
    m1 = {1: '01', 2: '04', 3: '07', 4: '10'}[q]
    d1 = '01'
    m2 = {1: '03', 2: '06', 3: '09', 4: '12'}[q]
    d2 = {3: '31', 6: '30', 9: '30', 12: '31'}[int(m2)]
    start = joiner.join([str(y), m1, d1])
    end = joiner.join([str(y), m2, d2])
    return start, end


def get_quarter_start_end_by_date(date=None):
    '''获取date日期所在季度的起始日期'''
    y, q = get_quarter(date, None)
    return copy_format(get_quarter_start_end_by_yq(y, q),
                       date)


def today_quarter(joiner='Q'):
    '''获取今日所在季度'''
    t = pd.to_datetime(datetime.datetime.now())
    y, q = t.year, t.quarter
    if not joiner:
        return (y, q)
    return joiner.join([str(y), str(q)])


def get_2part_next(part1, part2, step, n=1):
    '''
    Example
    -------
    >>> get_2part_next(2023, 2, 4, 2) # 2023Q2往后推2个季度
    (2023, 4)
    >>> get_2part_next(2023, 4, 4, 5) # 2023Q4往后推5个季度
    (2025, 1)
    >>> get_2part_next(2023, 4, 12, 8) # 202304往后推8个月
    (2023, 12)
    >>> get_2part_next(2023, 9, 12, 10) # 202309往后推10个月
    (2024, 7)
    >>> get_2part_next(2023, 2, 4, -2) # 2023Q2往前推2个季度
    (2022, 4)
    >>> get_2part_next(2023, 4, 4, -5) # 2023Q4往前推5个季度
    (2022, 3)
    >>> get_2part_next(2023, 4, 12, -8) # 202304往前推8个月
    (2022, 4)
    >>> get_2part_next(2023, 9, 12, -10) # 202309往前推10个月
    (2022, 11)
    '''
    assert all([isinstance(x, int) for x in [part1, part2, step, n]])
    if n == 0:
        p1new, p2new = part1, part2
    elif n > 0:
        p1add = n // step + int((part2 + n % step) > step)
        p1new = part1 + p1add
        p2new = part2 + (n % step)
        if p2new > step:
            p2new = p2new - step
    else:
        n = abs(n)
        p1add = -(n // step) - int((part2 - n % step) <= 0)
        p1new = part1 + p1add
        p2new = part2 - (n % step)
        if p2new <= 0:
            p2new = step + p2new
    return (p1new, p2new)


def get_pre_quarter(date=None, n=1, joiner='Q'):
    '''
    | 获取date前第n个季度
    | 如:
    | get_pre_quarter("20230418") -> "2023Q1"
    | get_pre_quarter("20230418", 2) -> "2022Q4"
    | get_pre_quarter("20230418", 3) -> "2022Q3"
    | get_pre_quarter("20230418", 4) -> "2022Q2"
    | get_pre_quarter("20230418", 5) -> "2022Q1"
    | get_pre_quarter("20230418", 6) -> "2021Q4"
    | get_pre_quarter("20230418", 7) -> "2021Q3"
    | get_pre_quarter("20230418", 8) -> "2021Q2"
    '''
    if pd.isnull(date):
        date = datetime.datetime.now()
    t = x2datetime(date)
    y, q = t.year, t.quarter
    ynew, qnew = get_2part_next(y, q, 4, -n)
    if not joiner:
        return (ynew, qnew)
    return joiner.join([str(ynew), str(qnew)])


def get_dayofweek(date=None):
    '''返回date属于星期几（1-7）'''
    if pd.isnull(date):
        date = datetime.date.today()
    return x2datetime(date).weekday() + 1


def get_dayofyear(date=None):
    '''返回date属于一年当中的第几天（从1开始记）'''
    if pd.isnull(date):
        date = datetime.date.today()
    return x2datetime(date).dayofyear


def get_month_end(date=None):
    '''获取date所在月的月末日期'''
    if pd.isnull(date):
        date = today_date()
    day = x2datetime(date)
    m = day.month
    if m != 2:
        month_end = day.replace(day=MONTH_DAYS[m])
    else:
        next_month = day.replace(day=28) + datetime.timedelta(days=4)
        month_end = next_month - datetime.timedelta(days=next_month.day)
    return copy_format(month_end, date)


def get_next_nmonth_start_end(date=None, n=1):
    if pd.isnull(date):
        date = today_date()
    day = x2datetime(date)
    y, m = day.year, day.month
    y, m = get_2part_next(y, m, 12, n)
    month_start = pd.to_datetime('%s-%s-01'%(y, m))
    next_month = month_start.replace(day=28) + datetime.timedelta(days=4)
    month_end = next_month - datetime.timedelta(days=next_month.day)
    return copy_format(month_start, date), copy_format(month_end, date)


def get_next_nquarter_start_end(date=None, n=1):
    if pd.isnull(date):
        date = today_date()
    day = x2datetime(date)
    y, q = day.year, day.quarter
    y, q = get_2part_next(y, q, 4, n)
    start, end = get_quarter_start_end_by_yq(y, q, '')
    return copy_format(start, date), copy_format(end, date)


def get_date_format(date,
                    joiners=[' ', '-', '/', '*', '#', '@', '.', '_']
                    ):
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
    
    assert isinstance(date, str), '只支持string格式日期'

    for joiner in joiners:
        reg = re.compile(r'\d{4}['+joiner+']\d{2}['+joiner+']\d{2}')
        if len(reg.findall(date)) > 0:
            return len(date), joiner

    # 只要8位都是数字就认为date是形如'20200226'格式的日期
    if len(date) == 8:
        tmp = [str(x) for x in range(0, 10)]
        tmp = [x in tmp for x in date]
        if all(tmp):
            return 8, ''

    return '未知日期格式', None


def date_reformat_simple(date, joiner_ori, joiner_new='-'):
    '''时间格式简单转化'''
    return joiner_new.join([x.zfill(2) for x in date.split(joiner_ori)])


def date_reformat_chn(date, joiner='-'):
    '''
    指定连接符为joiner，重新格式化date，date格式为x年x月x日
    '''
    res = date.replace('年', '-').replace('月', '-').replace('日', '')
    return date_reformat_simple(res, '-', joiner_new=joiner)


def gen_strformat(joiner=''):
    return '%Y{x}%m{x}%d'.format(x=joiner)


def date_reformat(date, joiner='-'):
    '''指定连接符为joiner，重新格式化date'''
    date_ = x2datetime(date)
    fmt = gen_strformat(joiner)
    return date_.strftime(fmt)
    

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


def diff_time_second(time1, time2):
    '''
    | 计算两个时间的差：time1-time2，time1和time2的格式应相同
    | 返回两个时间差的秒数
    '''
    time1_ = x2datetime(time1)
    time2_ = x2datetime(time2)
    time_delta = time1_ - time2_
    return 24*3600*time_delta.days + time_delta.seconds


def diff_days_date(date1, date2):
    '''
    计算两个日期间相隔天数，若date1大于date2，则输出为正，否则为负
    '''
    date1_ = x2datetime(date1)
    date2_ = x2datetime(date2)
    return (date1_-date2_).days


def get_dates_between(date1, date2, keep1=False, keep2=True,
                      only_workday=False, del_weekend=False,
                      joiner=2):
    '''
    获取date1到date2之间的日期列表，keep1和keep2设置结果是否保留date1和date2


    .. note:: 是否为workday用了chncal包，若chncal库没更新，可能导致结果不准确
    '''
    dates = pd.date_range(x2datetime(date1), x2datetime(date2)).tolist()
    if only_workday:
        dates = [x for x in dates if chncal.is_workday(x)]
    if del_weekend:
        dates = [x for x in dates if x.weekday() not in [5, 6]]
    if not keep1:
        dates = dates[1:]
    if not keep2:
        dates = dates[:-1]
    dates = copy_format(dates, date2) if joiner == 2 else copy_format(dates, date1)
    return dates


def get_work_dates_chncal(start_date, end_date=None):
    '''利用chncal获取指定范围内的工作日列表'''
    if pd.isnull(end_date):
        end_date = datetime.date.today()
    start_date_ = x2datetime(start_date)
    end_date_ = x2datetime(end_date)
    dates = chncal.get_workdays(start_date_, end_date_)
    dates = copy_format(dates, start_date)
    return dates


def get_recent_workday_chncal(date=None, dirt='post'):
    '''
    若date为工作日，则返回，否则返回下一个(post)或上一个(pre)工作日


    .. note:: 若chncal库没更新，可能导致结果不准确
    '''
    if pd.isnull(date):
        date = today_date()
    if dirt == 'post':
        while not chncal.is_workday(date):
            date = date_add_nday(date, 1)
    elif dirt == 'pre':
        while not chncal.is_workday(date):
            date = date_add_nday(date, -1)
    return date


def get_next_nth_workday_chncal(date=None, n=1):
    '''
    | 给定日期date，返回其后第n个工作日日期，n可为负数（返回结果在date之前）
    | 若n为0，直接返回date
    
    
    .. note:: 若chncal库没更新，可能导致结果不准确
    '''
    if pd.isnull(date):
        date = today_date()
    n_add = -1 if n < 0 else 1
    n = abs(n)
    tmp = 0
    while tmp < n:
        date = date_add_nday(date, n_add)
        if chncal.is_workday(date):
            tmp += 1
    return date


def get_date_weekday(weekday, n_week=0, joiner=''):
    '''
    获取指定星期的日期
    
    Parameters
    ----------
    weekday : int
        指定星期，如5表示星期5，7表示星期天
    n_week : int
        表示距离现在第几个星期，0表示当前星期，大于0表示往后推，小于0表示往前推
    
    Examples
    --------
    >>> get_date_weekday(3) # 获取本周三的日期
    >>> get_date_weekday(6, -1) # 获取上周六的日期
    >>> get_date_weekday(7, -1) # 获取下周日的日期
    '''
    assert weekday in range(1, 8)
    assert isinstance(joiner, (type(None), str))
    today = datetime.datetime.now()
    today_week = today.weekday() + 1
    diff_today = weekday - today_week
    delta = datetime.timedelta(days=diff_today+7*n_week)
    tgt_date = today + delta
    tgt_date = tgt_date.date()
    if not joiner is None:
        tgt_date = tgt_date.strftime(gen_strformat(joiner))
    return tgt_date


def get_recent_inweekday(date=None, dirt='post'):
    '''
    若date不是周末，则返回date，否则返回下一个(post)或上一个(pre)周内日期
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
    计算两个日期间相隔的工作日天数，若date1大于date2，则输出为正，否则为负
    '''
    raise NotImplementedError
