# -*- coding: utf-8 -*-

import requests


def gptchat0(prompt, api_key, **kwargs):
    url = 'http://openai.51sikao.cn:80/gptchat0'
    params = {}
    payload = {
        "question": prompt,
        "api_key": api_key
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


if __name__ == '__main__':
    api_key = 'MVBRFveFh-zRtEcQnc4x6ABdm47lKK4XVqjMZsQ77Do'
    
    # 单句对话
    res = gptchat0('中国有多少人', api_key=api_key)
    print(res)
    
    
    # 多句对话
    messages = [{"role": "user", "content": "中国有多少个省"},
                {'role': 'assistant', 'content': '中国共有34个省级行政区，包括23个省、5个自治区、4个直辖市和2个特别行政区。'},
                {"role": "user", "content": "地级市有多少个呢"}]
    res = gptchat0(messages,
                   api_key=api_key)
    print(res)