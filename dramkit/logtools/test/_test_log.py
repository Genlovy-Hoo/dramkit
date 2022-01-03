# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import time
    import _test_func
    from dramkit.logtools import utils_logger
    from dramkit.logtools import logger_general
    from dramkit.logtools import logger_rotating
    from dramkit.logtools import logger_timedrotating

    #%%
    logger = logger_general.get_logger(fpath='./log_test1.log',
                                       fmode='w',
                                       screen_show=True)

    logger.info('Log start here *********************************************')
    _test_func.test_func(3, 5, 'this is a warning.', logger)
    
    _test_func.test_func(3, 0, 'this is a error.', logger)
    logger.info('test finished.')

    utils_logger.close_log_file(logger)
    utils_logger.remove_handlers(logger)

    #%%
    logger = logger_rotating.get_logger(fpath='./log_test2.log',
                                        fmode='w', max_kb=1, nfile=3,
                                        screen_show=True)
    logger.info('Log start here *********************************************')
    _test_func.test_func(3, 1, 'this is a warning.', logger)
    logger.info('test finished.')

    utils_logger.close_log_file(logger)
    utils_logger.remove_handlers(logger)

    #%%
    log_path = './log_test3.log'
    logger = logger_timedrotating.get_logger(fpath=log_path, when='S',
                                             interval=3, nfile=3,
                                             screen_show=True)

    count = 0
    while count < 2:
        logger.info('{}th log start here ********************'.format(count+1))
        _test_func.test_func(3, 2, 'this is a warning.', logger)
        logger.info('test finished.')

        time.sleep(2)
        count += 1

    utils_logger.close_log_file(logger)
    utils_logger.close_log_file(logger)
