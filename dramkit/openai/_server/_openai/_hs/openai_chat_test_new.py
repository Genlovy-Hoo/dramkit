# -*- coding: utf-8 -*-
"""
@author: huyy43273
"""

import requests
from dramkit.gentools import tmprint


# 服务rul 
SERV_URL = 'http://20.243.132.35:8080'
SERV_URL = 'http://10.0.0.4:8080'
SERV_URL = 'http://openai.51sikao.cn:80'
#SERV_URL = 'http://127.0.0.1:80'


# 单句对话
def gptchat0(prompt, api_key=None, ft=None,
             **kwargs):
    url = '{}/gptchat0'.format(SERV_URL)
    params = {
        "ft": ft,
	}
    payload = {
        "question": prompt,
        "api_key": api_key
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


# 联系上下文对话
def gptchat(prompt, reset=0, api_key=None, ft=None,
            **kwargs):
    url = '{}/gptchat'.format(SERV_URL)
    params = {
        "ft": ft,
        "reset": reset
	}
    payload = {
        "question": prompt,
        "api_key": api_key
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


# 联系上下文对话
def openaichat(prompt, **kwargs):
    url = '{}/openaichat'.format(SERV_URL)
    params = {}
    payload = {"prompt": prompt}
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload,
                            params=params)
    return res.json()


if __name__ == '__main__':
    # api_key = 'MVBRFveFh-zRtEcQnc4x6ABdm47lKK4XVqjMZsQ77Do'
    # res = gptchat0('中国有多少人', api_key=api_key)
    # tmprint(res)
    # res = gptchat('印度有多少人', api_key=api_key, temperature=0.8)
    # tmprint(res)
    res = openaichat('你好', temperature=0.8)
    tmprint(res)
    
    # a1 = gptchat('写一段python代码实现对列表中的每个元素计数')
    # tmprint(a1['data']['result'])
    # a2 = gptchat('写得太复杂了，简化一点')
    # tmprint(a2['data']['result'])
    # a1 = gptchat('杭州房价怎么样', frequency_penalty=0.45)
    # tmprint(a1['data']['result'])
    # a2 = gptchat('相比于上海呢', presence_penalty=0.57)
    # tmprint(a2['data']['result'])

    # a1 = gptchat('杭州房价怎么样')
    # tmprint(a1['data']['result'])
    # a2 = gptchat('相比于北京呢', reset=1)
    # tmprint(a2['data']['result'])
    
    # res1 = gptchat0('恒生电子股份有限公司的证券代码是？')
    # tmprint(res1)
    # res2 = gptchat0('恒生电子股份有限公司的证券代码是？', ft='')
    # tmprint(res2)
    # res3 = gptchat0('恒生电子股份有限公司的证券代码是？', ft='default')
    # tmprint(res3)
    # res4 = gptchat0('恒生电子股份有限公司的证券代码是？', ft='nomodel')
    # tmprint(res4)
    
    # res1 = gptchat('科盾科技股份有限公司的中文简称？')
    # tmprint(res1)
    # res2 = gptchat('科盾科技股份有限公司的中文简称？', ft='')
    # tmprint(res2)
    # res3 = gptchat('科盾科技股份有限公司的中文简称？', ft='default')
    # tmprint(res3)
    # res4 = gptchat('科盾科技股份有限公司的中文简称？', ft='nomodel')
    # tmprint(res4)



    
    
    
    
    
    
