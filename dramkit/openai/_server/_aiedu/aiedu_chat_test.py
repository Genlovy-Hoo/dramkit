# -*- coding: utf-8 -*-
"""
@author: huyy43273
"""

import requests
from dramkit.gentools import tmprint


# 服务rul 
SERV_URL = 'http://localhost:8080'


# 单句对话
def gptchat0(prompt, **kwargs):
    url = '{}/gptchat0'.format(SERV_URL)
    params = {}
    payload = {
        "question": prompt
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


# 联系上下文对话
def gptchat(prompt, reset=0, *kwargs):
    url = '{}/gptchat'.format(SERV_URL)
    params = {
        "reset": reset
	}
    payload = {
        "question": prompt
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


if __name__ == '__main__':
    res = gptchat0('你真傻')
    tmprint(res)
    res = gptchat('秦始皇叫啥名字')
    tmprint(res)
    
    



    
    
    
    
    
    
