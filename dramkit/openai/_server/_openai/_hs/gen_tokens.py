# -*- coding: utf-8 -*-

import random
from jose import jwt
from dramkit.iotools import write_json


def gen_token(data):
    token = jwt.encode(data,
                       key='testkeys',
                       algorithm='HS256')
    return token


if __name__ == '__main__':
    n = 40 
    uinfos = [{'name': 'user_%s'%x,
               'pwd': 'pwd_%s'%x} for x in range(n)]
    tokens1 = {gen_token(x).split('.')[-1]: 5 for x in uinfos[:int(n/2)]}
    tokens2 = {gen_token(x).split('.')[-1]: 10 for x in uinfos[int(n/2):]}
    # write_json(tokens1, './tokens/tokens1.json')
    # write_json(tokens2, './tokens/tokens2.json')
    
    supers = [{'name': 'super_user_%s'%x,
               'pwd': 'super_pwd_%s'%x} for x in range(10)]
    tokens3 = {gen_token(x).split('.')[-1]: 100 for x in supers}
    # write_json(tokens3, './tokens/tokens3.json')
    
    nolimits = [{'name': 'nolimit_user_%s'%x,
                 'info': 'nolimit_info_%s'%x} for x in range(20)]
    tokens4 = {gen_token(x).split('.')[-1]: 0 for x in nolimits}
    # write_json(tokens4, './tokens/tokens4.json')
