# -*- coding: utf-8 -*-

# logging使用常用注意事项

if __name__ == '__main__':
    import time
    import logging
    
    #%%
    # 1. 新建同名logger时若之前已存在的logger里面handlers没移除，
    #    会导致重复打印日志
    def get_logger1(name=None):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        log.addHandler(h)
        return log
    
    # 第一次
    log1 = get_logger1('log1')
    log1.info('test log1 1')
    # 第二次
    print('\n')
    time.sleep(0.5)
    log1 = get_logger1('log1')
    log1.info('test log1 2')
    
    def get_logger2(name=None):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.handlers = []
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        log.addHandler(h)
        return log
    
    # 第三次
    print('\n')
    time.sleep(0.5)
    log1 = get_logger2('log1')
    log1.info('test log1 3')
    
    # 若新建logger与已有logger不同名，则不会出现重复打印的情况
    # 第一次
    print('\n')
    time.sleep(0.5)
    log1 = get_logger1('log2')
    log1.info('test log1 4')
    # 第二次
    print('\n')
    time.sleep(0.5)
    log1 = get_logger1('log3')
    log1.info('test log1 5')
    
    #%%
    # 2. 当新定义的logger的name与已存在的logger的name相同时，
    #    已存在logger的handler会被替换，
    #    因此在需要将内容写入不同日志文件时需要注意这个问题。
    def get_logger3(fpath, name=None):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.handlers = []
        h = logging.FileHandler(fpath, mode='w')
        h.setLevel(logging.DEBUG)
        log.addHandler(h)
        return log
    
    print('\n')
    time.sleep(0.5)
    log1 = get_logger3('./log1', 'a')
    print(log1.handlers)
    log2 = get_logger3('./log2', 'a')
    print(log1.handlers)
    print(log2.handlers)




