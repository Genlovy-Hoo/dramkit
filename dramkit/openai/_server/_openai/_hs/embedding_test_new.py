# -*- coding: utf-8 -*-

import requests


def embedding(prompt, api_key, **kwargs):
    url = 'http://openai.51sikao.cn:80/embedding'
    params = {}
    payload = {
        "prompt": prompt,
        "api_key": api_key
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


if __name__ == '__main__':
    # api_key = 'gKdpuwSPfVojx0UV-bRfZzjj-zO4YKBitTNpp-heEeo'
    api_key = 'DdyG5rIjgco6tOyOipwbQvQjhet6ahTkwokRZflpTbA'
    
    # embedding
    res = embedding('你好', api_key=api_key)
    print(res)
    