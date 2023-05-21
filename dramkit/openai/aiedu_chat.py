# -*- coding: utf-8 -*-

import os
import warnings
import requests
from dramkit.gentools import (try_repeat_run,
                              tmprint,
                              isnull)
from dramkit.iotools import (load_json,
                             write_json,
                             make_file_dir)
from dramkit.datetimetools import datetime_now


QUIT_WORDS = ['quit', 'bye', 'good bye', 'goodbye', '拜拜', '再见']
N_MAX_CON = 10 # 连续对话信息保留的最大轮数
N_TRY = 1 # 重试次数


@try_repeat_run(N_TRY)
def _chat(messages, **kwargs):
    '''
    | https://chat.forchange.cn/
    | https://www.aigcfun.com/
    '''
    url = 'https://api.aioschat.com/'
    payload = {'messages': messages,
               'model': 'gpt-3.5-turbo'
               }
    payload.update(kwargs)
    headers = {'content-type': 'application/json',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'}
    try:
        warnings.filterwarnings('ignore')
        response = requests.request('POST', url,
                                    json=payload,
                                    headers=headers,
                                    verify=False)
        warnings.filterwarnings('default')
        # res = response.json()['choices'][0]['message']['content'].strip()
        res = response.json()['choices'][0]['text'].strip()
        response.close()
        return True, res
    except Exception as exc:
        return False, exc.error.to_dict()
    
    
def _save2json(data, fpath):
    make_file_dir(fpath)
    if not os.path.exists(fpath):
        data = [data]
    else:
        data = load_json(fpath) + [data]
    write_json(data, fpath)


def chat(question, save_path=None, **kwargs):
    if not isnull(save_path):
        to_write = locals()
        to_write.pop('save_path')
    messages = [{'role': 'user', 'content': question}]
    ans = _chat(messages, **kwargs)
    if not isnull(save_path):
        to_write.update({'tm': datetime_now(),
                         'answer': ans})
        _save2json(to_write, save_path)
    return ans


def chat_con(**kwargs):
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
            suscess, result = _chat(messages, **kwargs)
            if not suscess:
                return turns, result
            turns += [question] + [result]
            print('\nAI answer:\n' + result)
            if result == '非常抱歉，根据我们的产品规则，无法为你提供该问题的回答，请尝试其他问题。':
                messages.pop()
            else:
                # 为了防止超过字数限制程序会爆掉，限制保留的对话次数
                if len(turns) > N_MAX_CON:
                    messages.pop(0)
                messages.append({'role': 'assistant', 'content': result})
            
            
class OpenAIChat(object):
    
    def __init__(self, **kwargs):
        self.messages = []
        self.turns = []
        
    def chat(self, question, save_path=None, **kwargs):
        if not isnull(save_path):
            to_write = locals()
            to_write.pop('self')
            to_write.pop('save_path')
        ans = self._chat(question, **kwargs)
        if not isnull(save_path):
            to_write.update({'tm': datetime_now(),
                             'answer': ans})
            _save2json(to_write, save_path)
        return ans
        
    def _chat(self, question, **kwargs):
        if len(question.strip()) == 0:
            print('AI: No correct question detected!')
            return 'No correct question detected!', {'error_info': 'No correct question detected!'}
        elif question.lower() in QUIT_WORDS:
            print('AI: See you next time!')
            return 'See you next time!', None
        else:
            messages = self.messages + [{'role': 'user', 'content': question}]
            suscess, result = _chat(messages, **kwargs)
            if not suscess:
                return '', result
            self.turns += [question] + [result]
            if not result == '非常抱歉，根据我们的产品规则，无法为你提供该问题的回答，请尝试其他问题。':
                if len(self.turns) > N_MAX_CON:
                    self.messages.pop(0)
                self.messages.append({'role': 'assistant', 'content': result})
            return result, None


if __name__ == '__main__':
    save_path = './_test/test_aiedu.json'
    
    question = '中国有多少个地级市'
    # question = '中国人口最多的是哪个省'
    ans = chat(question, save_path=save_path)
    tmprint(ans)
    
    # turns = chat_con()
    
    chater = OpenAIChat()
    ans1 = chater.chat('南京房价如何', save_path=save_path)
    tmprint(ans1)
    # ans2 = chater.chat('相比于上海呢')
    # tmprint(ans2)
    
    
    
    
    
    