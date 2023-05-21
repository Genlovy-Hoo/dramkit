# -*- coding: utf-8 -*-

import os
import requests
import numpy as np


SERV_URL = 'http://openai.51sikao.cn:80'
PRIVATE_KEY = 'VVVqm3WyHlrsXVqoghaHWFDmJJ7F3_yo8uf1I5TYZ8Q'

QUIT_WORDS = ['quit', 'bye', 'good bye', 'goodbye', '拜拜', '再见']

N_MAX_CON = 6 # 连续对话信息保留的最大轮数
MAX_PROMPT_LEN = 1500


class ContinuousDialogueError(Exception):
    pass


def _set_con_chat():
    os.environ['OPENAI_CON_CHAT'] = '1'
    
    
_set_con_chat()


def raise_con():
    if not os.environ.get('OPENAI_CON_CHAT'):
        raise ContinuousDialogueError(
              'ᵔ◡ᵔ为了节省流量，默认不支持连续对话(•◡•)，请使用chat接口进行单句对话！')
    else:
        return


class DeprecationError(Exception):
    pass


class OpenAIChat(object):
    def __init__(self, *args, **kwargs):
        raise DeprecationError('该接口已弃用，联系上下文对话请使用chat_mem接口！')


# 单句对话
def chat(prompt, **kwargs):
    url = '{}/gptchat0'.format(SERV_URL)
    params = {}
    payload = {
        "question": prompt,
        "api_key": PRIVATE_KEY
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


# 联系上下文对话
def chat_mem(prompt, reset=0, **kwargs):
    raise_con()
    url = '{}/gptchat'.format(SERV_URL)
    params = {"reset": reset}
    payload = {
        "question": prompt,
        "api_key": PRIVATE_KEY
        }
    payload.update(kwargs)
    res = requests.request("POST", url,
                            json=payload, params=params)
    return res.json()


# 交互式连续对话
def chat_con(**kwargs):
    raise_con()
    print('Yon can input `quit` to stop the conversation.')
    messages = [] # 保存上文环境
    turns = [] # 保存连续对话记录
    while True: # 能够连续提问
        question = input('\nPlease input your question:\n')
        if len(question.strip()) == 0:
            # 如果输入为空，提醒输入问题
            print('Please input your question:')
        # 如果输入为'quit'，程序终止
        elif question.lower() in QUIT_WORDS:
            print('\nAI: See you next time!')
            return turns, None
        else:
            messages.append({'role': 'user', 'content': question})
            res = chat(messages, **kwargs)
            suscess, result = res['data']['suscess'], res['data']['result']
            if not suscess:
                return turns, result
            turns += [question] + [result]
            print('\nAI answer:\n' + result)
            # 为了防止超过字数限制程序会爆掉，限制保留的对话次数
            if len(turns) > N_MAX_CON:
                messages.pop(0)
            messages.append({'role': 'assistant', 'content': result})
            def _cut(messages):
                lens = [len(x['content']) for x in messages]
                cumsumlens = np.cumsum(lens[::-1])
                idx, len_ = 0, cumsumlens[0]
                while len_ < MAX_PROMPT_LEN and idx < len(lens)-1:
                    idx += 1
                    len_ = cumsumlens[idx]
                return messages[-idx:] # 超出那句不要
                # return messages[-(idx+1):] # 保留超出那句
            messages = _cut(messages)


if __name__ == '__main__':
    from dramkit import tmprint
    
    '''
    # chat单句对话
    res = chat('印度有多少人')
    tmprint(res)
    # '''
    
    '''
    # chat多句对话
    messages = [{"role": "user", "content": "中国有多少个省"},
                {'role': 'assistant', 'content': '中国共有34个省级行政区，包括23个省、5个自治区、4个直辖市和2个特别行政区。'},
                {"role": "user", "content": "地级市有多少个呢"}]
    res = chat(messages)
    tmprint(res)
    '''
    
    '''
    # chat_mem联系上下文对话
    ans1 = chat_mem('杭州房价怎么样', frequency_penalty=0.45)
    tmprint(ans1)
    ans2 = chat_mem('相比于上海呢', presence_penalty=0.57)
    tmprint(ans2)
    
    ans3 = chat_mem('南京房价怎么样')
    tmprint(ans3)
    ans4 = chat_mem('相比于北京呢', reset=1)
    tmprint(ans4)
    
    # '''
    
    
    
    
    
    
    
    