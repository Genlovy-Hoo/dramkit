# -*- coding: utf-8 -*-

# 参考：
# http://www.ncqh.cn/news/19723.html
# https://www.bilibili.com/read/cv21728710

# TODO: token占用数量改为取openai官方返回中的'usage'字段
# (目前直接用len()误差大)


import os
import openai
import requests
import numpy as np
from bs4 import BeautifulSoup
from dramkit.gentools import isnull, try_repeat_run, tmprint
from dramkit.iotools import load_yml, get_parent_path, load_json, write_json, make_file_dir
from dramkit.datetimetools import datetime_now
from dramkit.cryptotools import en_aes_cbc, de_aes_cbc


API_KEY = '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'

QUIT_WORDS = ['quit',
              'exit',
              'bye',
              'good bye',
              'goodbye',
              '拜拜',
              '再见',
              '退出',
              '结束',
              '结束对话']
N_TRY = 1 # 重试次数

CON = True # 是否允许连续对话
N_MAX_CON = 20 # 连续对话信息保留的最大轮数
MAX_PROMPT_LEN = 1500


def load_config():
    cfg = {}
    cfg_dir = get_parent_path(os.path.dirname(__file__), 2)
    fcfg = os.path.join(cfg_dir, 'config/openai.yml')
    if os.path.exists(fcfg):
        cfg = load_yml(fcfg)
    if not 'secret_key' in cfg:
        cfg['secret_key'] = 'openaiapikeykey'
    if not 'secret_iv' in cfg:
        cfg['secret_iv'] = 'openaiapikeyiv'
    return cfg
CFG = load_config()


class ContinuousDialogueError(Exception):
    pass


def raise_con():
    if not CON:
        raise ContinuousDialogueError('ᵔ◡ᵔ为了节省流量，默认不支持连续对话(•◡•)')
    else:
        return


def secret(api_key):
    return en_aes_cbc(api_key, key=CFG['secret_key'], iv=CFG['secret_iv'])


def unsecret(secret_api_key):
    return de_aes_cbc(secret_api_key, key=CFG['secret_key'], iv=CFG['secret_iv'])


def get_api_key_url():
    try:
        url = 'http://www.glhyy.cn/Other/mykey.html'
        html = requests.get(url, headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
                })
        bsobj = BeautifulSoup(html.content, 'lxml')
        api_key = bsobj.find('p').get_text().strip()
        return unsecret(api_key).strip()
    except:
        return None


def get_api_key():
    if not 'api_key' in CFG:
        api_key = get_api_key_url()
        if not isnull(api_key):
            CFG['api_key'] = api_key
    if 'api_key' in CFG:
        return CFG['api_key']
    else:
        return API_KEY
    
    
API_KEY_DEFAULT = get_api_key()


def _set_api_key(api_key):
    if isnull(api_key):
        api_key = API_KEY_DEFAULT
    openai.api_key = api_key
    
    
# embedding可用模型
EMBD_MODELS = [
    'babbage-code-search-code',
    'text-similarity-babbage-001',
    'babbage-code-search-text',
    'babbage-similarity',
    'code-search-babbage-text-001',
    'code-search-babbage-code-001',
    'text-embedding-ada-002',
    'text-similarity-ada-001',
    'ada-code-search-code',
    'ada-similarity',
    'code-search-ada-text-001',
    'text-search-ada-query-001',
    'davinci-search-document',
    'ada-code-search-text',
    'text-search-ada-doc-001',
    'text-similarity-curie-001',
    'code-search-ada-code-001',
    'ada-search-query',
    'text-search-davinci-query-001',
    'curie-search-query',
    'davinci-search-query',
    'babbage-search-document',
    'ada-search-document',
    'text-search-curie-query-001',
    'text-search-babbage-doc-001',
    'curie-search-document',
    'text-search-curie-doc-001',
    'babbage-search-query',
    'text-search-davinci-doc-001',
    'text-search-babbage-query-001',
    'curie-similarity',
    'text-similarity-davinci-001',
    'davinci-similarity'
]


@try_repeat_run(N_TRY)
def _get_embedding(prompt, api_key=None, **kwargs):
    '''OpenAI Embedding'''
    _set_api_key(api_key)
    params = {
        'model': 'text-embedding-ada-002',
        'input': prompt,
        }
    params.update(kwargs)
    try:
        response = openai.Embedding.create(**params)
        res = response['data'][0]['embedding']
        return True, res
    except Exception as exc:
        return False, exc.error.to_dict()


def get_embedding(prompt, api_key=None,
                  save_path=None, save_embd=False,
                  write_api_key=True,
                  **kwargs):
    '''OpenAI Embedding'''
    if not isnull(save_path):
        to_write = locals()
        to_write.pop('save_path')
        if ('api_key' in to_write) and (not write_api_key):
            to_write.pop('api_key')
    embd = _get_embedding(prompt, api_key=api_key, **kwargs)
    if not isnull(save_path):
        to_write.update({'tm': datetime_now(),
                         'suscess': embd[0]})
        if save_embd:
            to_write.update({'embd': embd[1]})
        _save2json(to_write, save_path)
    return embd


@try_repeat_run(N_TRY)
def chat_gpt3(prompt, api_key=None, **kwargs):
    '''
    | OpenAI GPT3模型单句对话
    | 返回(True/False, 回答内容/错误信息)
    '''
    _set_api_key(api_key)
    params = {
        'model': 'text-davinci-003',
        'prompt': prompt,
        'temperature': 1,
        'max_tokens': 2500,
        'top_p': 1,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
        }
    params.update(kwargs)
    try:
        response = openai.Completion.create(**params)
        answer = response['choices'][0]['text'].strip()
        return True, answer
    except Exception as exc:
        return False, exc.error.to_dict()
    
    
@try_repeat_run(N_TRY)
def _chat_gpt35(messages, api_key=None, **kwargs):
    '''
    | OpenAI GPT3.5模型(ChatGPT)单句对话
    | 返回(True/False, 回答内容/错误信息)
    '''
    _set_api_key(api_key)
    params = {
        'model': 'gpt-3.5-turbo',
        'messages': messages,
        'temperature': 1, # 控制结果随机性，0.0表示结果固定，越大随机性越大
        'max_tokens': 2500, # 最大返回字数（包括问题和答案），通常汉字占两个token。假设设置成100，如果prompt问题中有40个汉字，那么返回结果中最多包括10个汉字。ChatGPT API gpt-3.5-turbo模型允许的最大token数量为4096，即max_tokens最大设置为4096减去问题的token数量
        'top_p': 1,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'stream': False
        }
    params.update(kwargs)
    try:
        response = openai.ChatCompletion.create(**params)
        answer = response.choices[0].message.content.strip()
        return True, answer
    except Exception as exc:
        return False, exc.error.to_dict()
    
    
@try_repeat_run(N_TRY)
def chat_gpt35(question, api_key=None, **kwargs):
    '''
    | OpenAI GPT3.5模型(ChatGPT)单句对话
    | 返回(True/False, 回答内容/错误信息)
    '''
    assert isinstance(question, (str, list))
    if isinstance(question, str):
        messages = [{'role': 'user', 'content': question}]
    else:
        messages = question
    return _chat_gpt35(messages, api_key=api_key, **kwargs)


def _save2json(data, fpath):
    make_file_dir(fpath)
    if not os.path.exists(fpath):
        data = [data]
    else:
        data = load_json(fpath) + [data]
    write_json(data, fpath)
    
    
def chat(prompt, api_key=None, version='gpt3.5',
         save_path=None, write_api_key=True, **kwargs):
    '''OpenAI GPT模型单句对话'''
    if not isnull(save_path):
        to_write = locals()
        to_write.pop('save_path')
    if ('api_key' in to_write) and (not write_api_key):
        to_write.pop('api_key')
    if version.lower() in ['gpt3.5', 'gptchat', 'chatgpt']:
        ans = chat_gpt35(prompt, api_key=api_key, **kwargs)
    elif version.lower() in ['gpt3']:
        ans = chat_gpt3(prompt, api_key=api_key, **kwargs)
    else:
        raise ValueError('未识别的版本：`{}`'.format(version))
    if not isnull(save_path):
        to_write.update({'tm': datetime_now(),
                         'answer': ans})
        _save2json(to_write, save_path)
    return ans


def chat_con_gpt3(api_key=None, **kwargs):
    '''gpt3交互式连续对话'''
    raise_con()
    print('Yon can input `quit` to stop the conversation.')
    prompts = '' # 保存上文环境
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
            prompts = prompts + '\nHuman: ' + question
            suscess, result = chat_gpt3(prompts, api_key=api_key, **kwargs)
            if not suscess:
                return turns, result
            turns += [question] + [result] # 只有这样迭代才能连续提问理解上下文
            print('\nAI answer:\n' + result)
            # 为了防止超过字数限制程序会爆掉，限制保留的对话次数
            if len(turns) <= N_MAX_CON:
                prompts = '\n'.join(turns)
            else:
                prompts = '\n'.join(turns[-N_MAX_CON:])
                
                
def chat_con_gpt35(api_key=None, **kwargs):
    '''gpt3.5(ChatGPT)交互式连续对话'''
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
            suscess, result = _chat_gpt35(messages, api_key=api_key, **kwargs)
            if not suscess:
                return turns, result
            turns += [question] + [result]
            print('\nAI answer:\n' + result)
            # 为了防止超过字数限制程序会爆掉，限制保留的对话次数
            if len(turns) > N_MAX_CON:
                messages.pop(0)
            messages.append({'role': 'assistant', 'content': result})
                
                
def chat_con(api_key=None, version='gpt3.5', **kwargs):
    '''OpenAI GPT模型连续对话'''
    if version.lower() in ['gpt3.5', 'gptchat', 'chatgpt']:
        return chat_con_gpt35(api_key=api_key, **kwargs)
    elif version.lower() in ['gpt3']:
        return chat_con_gpt3(api_key=api_key, **kwargs)
    else:
        raise ValueError('未识别的版本：`{}`'.format(version))
                
                
class OpenAIChat(object):
    
    def __init__(self, api_key=None):
        raise_con()
        self.api_key = api_key if not isnull(api_key) else API_KEY_DEFAULT
        self.prompts = ''
        self.messages = []
        self.turns = []
        
    def chat(self, question, reset=False, api_key=None, version='gpt3.5',
             save_path=None, write_api_key=True, **kwargs):
        if not isnull(save_path):
            to_write = locals()
            to_write.pop('self')
            to_write.pop('save_path')
        if ('api_key' in to_write) and (not write_api_key):
            to_write.pop('api_key')
        if version.lower() in ['gpt3.5', 'gptchat', 'chatgpt']:
            ans = self.chat_gpt35(question, reset=reset, api_key=api_key, **kwargs)
        elif version.lower() in ['gpt3']:
            ans = self.chat_gpt3(question, reset=reset, api_key=api_key, **kwargs)
        else:
            raise ValueError('未识别的版本：`{}`'.format(version))
        if not isnull(save_path):
            to_write.update({'tm': datetime_now(),
                             'answer': ans})
            _save2json(to_write, save_path)
        return ans
        
    def chat_gpt3(self, question, reset=False, api_key=None, **kwargs):
        # 如果输入为空或结束词，直接返回
        if len(question.strip()) == 0:
            print('AI: No correct question detected!')
            return 'No correct question detected!', {'error_info': 'No correct question detected!'}
        elif question.lower() in QUIT_WORDS:
            print('AI: See you next time!')
            return 'See you next time!', None
        else:
            if reset:
                self.prompts = ''
                self.turns = []
            prompt = self.prompts + '\nHuman: ' + question
            api_key_ = self.api_key if isnull(api_key) else api_key
            suscess, result = chat_gpt3(prompt, api_key=api_key_, **kwargs)
            if not suscess:
                return '', result
            self.turns += [question] + [result]
            if len(self.turns) <= N_MAX_CON:
                self.prompts = '\n'.join(self.turns)
            else:
                self.prompts = '\n'.join(self.turns[-N_MAX_CON:])
            if len(self.prompts) > MAX_PROMPT_LEN:
                self.prompts = self.prompts[-MAX_PROMPT_LEN:]
            return result, None
        
    def chat_gpt35(self, question, reset=False, api_key=None, **kwargs):
        if len(question.strip()) == 0:
            print('AI: No correct question detected!')
            return 'No correct question detected!', {'error_info': 'No correct question detected!'}
        elif question.lower() in QUIT_WORDS:
            print('AI: See you next time!')
            return 'See you next time!', None
        else:
            if reset:
                self.messages = []
                self.turns = []
            messages = self.messages + [{'role': 'user', 'content': question}]
            api_key_ = self.api_key if isnull(api_key) else api_key
            suscess, result = _chat_gpt35(messages, api_key=api_key_, **kwargs)
            if not suscess:
                return '', result
            self.turns += [question] + [result]            
            if len(self.turns) > N_MAX_CON:
                self.messages.pop(0)
            self.messages.append({'role': 'assistant', 'content': result})
            def _cut(messages):
                lens = [len(x['content']) for x in messages]
                cumsumlens = np.cumsum(lens[::-1])
                idx, len_ = 0, cumsumlens[0]
                while len_ < MAX_PROMPT_LEN and idx < len(lens)-1:
                    idx += 1
                    len_ = cumsumlens[idx]
                # return self.messages[-idx:] # 超出那句不要
                return self.messages[-(idx+1):] # 保留超出那句
            self.messages = _cut(self.messages)
            return result, None
                
                
if __name__ == '__main__':
    pass

    # ok, embd = get_embedding('你好')

    # ans1 = chat('中国有多少个地级市', version='gpt3')
    # tmprint(ans1)
    # ans2 = chat('中国有多少个县级市')
    # tmprint(ans2)
    
    
    # turns1, err1 = chat_con(version='gpt3')
    # turns2, err2 = chat_con()
    
    
    # chater1 = OpenAIChat(version='gpt3')
    # res11 = chater1.chat('南京房价如何')
    # tmprint(res11)
    # res12 = chater1.chat('相比于北京呢')
    # tmprint(res12)
    
    
    # chater2 = OpenAIChat()
    # res21 = chater2.chat('杭州房价如何')
    # tmprint(res21)
    # res22 = chater2.chat('相比于上海呢')
    # tmprint(res22)
    
    
    # chater3 = OpenAIChat()
    # res31 = chater3.chat('南京房价如何',
    #                      save_path='./_test/test.json')
    # tmprint(res31)
    # res32 = chater3.chat('相比于北京呢', reset=True, 
    #                      save_path='./_test/test.json')
    # tmprint(res32)
    
    # chater4 = OpenAIChat()
    # res41 = chater4.chat('写一篇300字左右的文章介绍西湖',
    #                      save_path='./_test/test.json')
    # tmprint(res41)
    # res42 = chater4.chat('再写一篇')
    # tmprint(res42)
    # res43 = chater4.chat('再写一篇500左右的文章介绍杭州')
    # tmprint(res43)
    # res44 = chater4.chat('再写一篇')
    # tmprint(res44)
    
    
    
    
    
    
    
    
    
