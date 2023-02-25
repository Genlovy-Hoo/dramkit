# -*- coding: utf-8 -*-

# 命令行使用 python _test_argparse.py -h查看参数帮助

import sys
from dramkit.gentools import (
                      parse_args,
                      gen_args_praser
                      )


def testmain(a, b:list, c=1, d:int=2):
    print(a, b, c, d)


if __name__ == '__main__':
    '''
    parser = parse_args([(['-e', '--epochs'], 
                          {'type': int, 'default': 30}),
                         (['-b', '--batch'],
                          {'type': int, 'default': 4,
                           'help': 'Batch'})],
                        description='test')
    args = parser.parse_args(sys.argv[1:])
    print(sys.argv, end='\n\n')
    print(args)
    # '''
    
    '''
    parser = parse_args({'group1': [(['-e', '--epochs'], 
                                     {'type': int, 'default': 30}),
                                    (['-b', '--batch'],
                                     {'type': int, 'default': 4,
                                      'help': 'Batch'}),
                                    # (['-a', '--apos'], 
                                    #  {'type': int,
                                    #   'required': True}),
                                    ],
                         'group2': [(['-lr', '--learning-rate'],
                                     {'type': float, 'default': 0.1,
                                      'help': 'lr'}),
                                    (['--number-tree'],
                                     {'type': int, 'default': 5,
                                      'help': 'number of tree'}),
                                    (['--p_q'],
                                     {'type': int, 'default': 0,
                                      'help': 'p&q'}),
                                    (['-s'],
                                     {'default': 0,
                                      'help': 'string'})]},
                        description='test')
    args = parser.parse_args(sys.argv[1:])
    # print(sys.argv[1:], end='\n\n')
    print(args)
    print(args.p_q)
    # '''
    
    # '''
    parser = gen_args_praser(testmain)
    args = parser.parse_args(sys.argv[1:])
    print(args, end='\n\n')
    print(args.a)
    testmain(args.a, args.b, args.c, args.d)
    # '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    