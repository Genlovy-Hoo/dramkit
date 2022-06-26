# -*- coding: utf-8 -*-

import numpy as np

def test_func(x, y, txt, logger):
    logger.info('info of test func ------------------------------------------')
    logger.info('get input x: {}, y: {}.'.format(x, y))
    logger.warning(txt)
    logger.error('some error may find here.')
    try:
        logger.debug('x div y : {}'.format(str(x/y)))
    except:
        logger.error('x cannot div y, please check y.', exc_info=True)
        # raise
    logger.fatal('program may crash here.')
    logger.info('test function finished.')
    
    
def rand_div(x):
    y = np.random.randint(-1, 1)
    try:
        return x / y
    except:
        raise 
