# -*- coding: utf-8 -*-
"""
@author: huyy43273
"""

import json
import requests


# 服务rul
SERV_URL = 'http://openai.51sikao.cn:80'


# 联系上下文对话
def openaichat0(prompt, **kwargs):
    url = '{}/openaichat'.format(SERV_URL)
    params = {}
    payload = {"prompt": prompt}
    payload.update(kwargs)
    res = requests.request("POST", url,
                           json=payload,
                           params=params)
    return res.json()


def openaichat(prompt, **kwargs):
    url = '{}/openaichat'.format(SERV_URL)
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    payload.update(kwargs)
    res = requests.request("POST", url,
                           headers=headers,
                           data=json.dumps(payload)
                           )
    return res.json()


if __name__ == '__main__':
    res = openaichat('你好', temperature=0.8)
    print(res)
    