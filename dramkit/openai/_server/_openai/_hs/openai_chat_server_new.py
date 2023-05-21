# -*- coding: utf-8 -*-

# 启动方式：
# 1. python main.py
# 2. uvicorn main:app --reload

import os
import time
import uvicorn
from typing import Optional, Union, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from dramkit.gentools import isnull
from dramkit.iotools import make_dir, load_json
from dramkit.openai.openai_chat import OpenAIChat, chat, get_embedding


# OPENAI_API_KEY = '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'
OPENAI_API_KEY = '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'
MODEL = 'gpt-3.5-turbo'

PRIVATE_KEY = 'AHg3cFvtxmDR-7hORbgNzgV29MBP7CV8f4VCj-vLdRo'


# async def _get_token(request: Request) -> str:
#     '''提取访问token'''
#     data = await request.json()
#     token = data['api_key']
#     return token


def get_token(request: Request) -> str:
    '''提取访问token'''
    data = request._json
    try:
        return data['api_key']
    except:
        return PRIVATE_KEY


TOKENS = {}
for fpath in os.listdir('./tokens/'):
    TOKENS.update(load_json('./tokens/{}'.format(fpath)))

WAIT_SECONDS = 60*10
global CONVERS
CONVERS = {} # {ip: {'last_tm': tm, 'chater': OpenAIChat()}}

# CONVERS_DIR = None
CONVERS_DIR = './conversations/'
if not isnull(CONVERS_DIR):
    make_dir(CONVERS_DIR)
    
SAVE_EMBD = False

FT_DEFAULT = 'davinci:ft-personal:hsgpt-2023-03-15-10-10-47'
FT_STOP = '\n'


app = FastAPI()

limiter = Limiter(key_func=get_token)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,
                          _rate_limit_exceeded_handler)


def clean_convers(CONVERS):
    res = {}
    for ip in CONVERS:
        if time.time()-CONVERS[ip]['last_tm'] <= WAIT_SECONDS:
            res[ip] = CONVERS[ip]
    return res


def _check_api_key(api_key):
    if api_key == '':
        api_key = None
    return api_key


def _check_ft(ft):
    if isinstance(ft, str) and (ft == '' or ft.lower() == 'default'):
        ft = FT_DEFAULT
    return ft


def _check_stop(stop):
    if stop == '':
        stop = FT_STOP
    return stop


class OpenaiParams(BaseModel):
    question: Union[str, List]
    api_key: str
    model: str = 'gpt-3.5-turbo'
    max_tokens: int = 2000
    temperature: float = 1
    top_p: float = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str = None
    
    
def get_limit(key: str):
    if not key in TOKENS:
        return '0/minute'
    if TOKENS[key] == 0:
        # return '100000/minute'
        return None
    return '{}/minute'.format(TOKENS[key])
        
        
# gptchat聊天接口，单句聊天
@app.post('/gptchat0')
@limiter.shared_limit(get_limit, 'app')
async def gptchat0(request: Request, 
                   params: Optional[OpenaiParams],
                   ft: Optional[str] = None
                   ):
    token = get_token(request)
    if token not in TOKENS:
        res = {'result': '', 'error_info': 'bad api_key!'}
        return JSONResponse(content={'data': res})
    params_ = params.dict().copy()
    prompt = params_.pop('question')
    params_.pop('api_key')
    params_.pop('model')
    ft = _check_ft(ft)
    save_path = None
    if not isnull(CONVERS_DIR):
        # TODO: 在linux服务器上这样取ip貌似时正确的，
        # 但是在windows服务上取到的ip是127.0.0.1，待解决
        ip = request.client.host
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    if not ft:
        suscess, result = chat(prompt,
                               api_key=OPENAI_API_KEY,
                               model=MODEL,
                               save_path=save_path,
                               write_api_key=False,
                               **params_)
        # suscess, result = True, 'test'
    else:
        del params_['max_tokens']
        del params_['model']
        stop = params_.pop('stop')
        stop = _check_stop(stop)
        suscess, result = chat(prompt,
                               api_key=OPENAI_API_KEY,
                               save_path=save_path,
                               version='gpt3',
                               model=ft,
                               max_tokens=800,
                               stop=stop,
                               write_api_key=False,
                               **params_)
        # isuscess, result = True, 'test'
    res = {'result': result, 'suscess': suscess}
    return JSONResponse(content={'data': res})
    

# gptchat聊天接口，连续聊天
@app.post('/gptchat')
@limiter.shared_limit(get_limit, 'app')
async def gptchat(request: Request,
                  params: Optional[OpenaiParams],
                  reset: Optional[bool] = False,
                  ft: Optional[str] = None
                  ):
    token = get_token(request)
    if token not in TOKENS:
        res = {'result': '', 'error_info': 'bad api_key!'}
        return JSONResponse(content={'data': res})
    params_ = params.dict().copy()
    prompt = params_.pop('question')
    params_.pop('api_key')
    params_.pop('model')
    ft = _check_ft(ft)
    ip = request.client.host
    tmnow = time.time()
    global CONVERS
    if ip not in CONVERS or tmnow-CONVERS[ip]['last_tm'] > WAIT_SECONDS or reset:
        CONVERS[ip] = {'last_tm': tmnow, 'chater': OpenAIChat()}
    CONVERS[ip]['last_tm'] = time.time()
    CONVERS = clean_convers(CONVERS)
    tmprint('The remaining IPs with conversation:', CONVERS.keys())
    save_path = None
    if not isnull(CONVERS_DIR):
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    if not ft:
        result, error_info = CONVERS[ip]['chater'].chat(
                              prompt,
                              api_key=OPENAI_API_KEY,
                              model=MODEL,
                              save_path=save_path,
                              write_api_key=False,
                              **params_)
        # result, error_info = 'test', None
    else:
        del params_['max_tokens']
        del params_['model']
        stop = params_.pop('stop')
        stop = _check_stop(stop)
        result, error_info = CONVERS[ip]['chater'].chat(
                                prompt,
                                api_key=OPENAI_API_KEY,
                                save_path=save_path,
                                version='gpt3',
                                model=ft,
                                max_tokens=800,
                                stop=stop,
                                write_api_key=False,
                                **params_)
        # result, error_info = 'test', None
    res = {'result': result, 'error_info': error_info}
    return JSONResponse(content={'data': res})


class OpenaiChatParams(BaseModel):
    prompt: Union[str, List]
    api_key: str = PRIVATE_KEY
    model: str = 'gpt-3.5-turbo'
    max_tokens: int = 2000
    temperature: float = 1
    top_p: float = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    reset: bool = 0
    stop: str = None


# gptchat聊天接口，连续聊天
@app.post('/openaichat')
@limiter.shared_limit(get_limit, 'app')
async def openaichat(request: Request,
                     params: Optional[OpenaiChatParams]
                     ):
    token = get_token(request)
    if token not in TOKENS:
        return JSONResponse(
                   content={'data': None,
                            'code': '01',
                            'msg': 'bad api_key'})
    params_ = params.dict().copy()
    prompt = params_.pop('prompt')
    reset = params_.pop('reset')
    params_.pop('api_key')
    ip = request.client.host
    tmnow = time.time()
    global CONVERS
    if ip not in CONVERS or tmnow-CONVERS[ip]['last_tm'] > WAIT_SECONDS or reset:
        CONVERS[ip] = {'last_tm': tmnow, 'chater': OpenAIChat()}
    CONVERS[ip]['last_tm'] = time.time()
    CONVERS = clean_convers(CONVERS)
    tmprint('The remaining IPs with conversation:', CONVERS.keys())
    save_path = None
    if not isnull(CONVERS_DIR):
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    result, error_info = CONVERS[ip]['chater'].chat(
                          prompt,
                          api_key=OPENAI_API_KEY,
                          save_path=save_path,
                          write_api_key=False,
                          **params_)
    # result, error_info = 'test', None
    if isnull(error_info):
        res = {'data': result, 'code': '00', 'msg': '成功'}
    elif result == 'No correct question detected!':
        res = {'data': None, 'code': '03', 'msg': 'bad prompt'}
    else:
        res = {'data': None, 'code': '02', 'msg': error_info}
    return JSONResponse(content=res)


class EmbdParams(BaseModel):
    prompt: str
    api_key: str
    model: str = 'text-embedding-ada-002'


# OpenAI Embedding
@app.post('/embedding')
@limiter.shared_limit(get_limit, 'app')
async def embedding(request: Request,
                    params: Optional[EmbdParams],
                    ):
    token = get_token(request)
    if token not in TOKENS:
        res = {'result': '', 'error_info': 'bad api_key!'}
        return JSONResponse(content={'data': res})
    params_ = params.dict().copy()
    prompt = params_.pop('prompt')
    params_.pop('api_key')
    save_path = None
    if not isnull(CONVERS_DIR):
        ip = request.client.host
        save_path = os.path.join(CONVERS_DIR, '%s_embd.json'%ip)
    suscess, result = get_embedding(prompt,
                                    api_key=OPENAI_API_KEY,
                                    save_path=save_path,
                                    save_embd=SAVE_EMBD,
                                    write_api_key=False,
                                    **params_)
    # suscess, result = True, [0.1, 0.2]
    res = {'result': result, 'suscess': suscess}
    return JSONResponse(content={'data': res})


from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from dramkit import tmprint

templates = Jinja2Templates(directory='templates')

# 添加首页
@app.get('/')
async def home(req: Request):
    return templates.TemplateResponse(
            'index.html',
            context={'request': req,
                     'ans': '',
                     'err': ''})


# 网页提交聊天
@app.post('/webchat')
async def webchat(req: Request, prompt: str = Form(None)):
    # print('question:')
    # tmprint(prompt)
    ip = req.client.host
    tmnow = time.time()
    global CONVERS
    if ip not in CONVERS or tmnow-CONVERS[ip]['last_tm'] > WAIT_SECONDS:
        CONVERS[ip] = {'last_tm': tmnow, 'chater': OpenAIChat()}
    CONVERS[ip]['last_tm'] = time.time()
    CONVERS = clean_convers(CONVERS)
    tmprint('The remaining IPs with conversation:', CONVERS.keys())
    save_path = None
    if not isnull(CONVERS_DIR):
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    result, error_info = CONVERS[ip]['chater'].chat(
                         prompt, save_path=save_path,
                         write_api_key=False)
    # print('answer:')
    # print(result)
    # print('error info:')
    # tmprint(error_info)
    return templates.TemplateResponse(
            'index.html',
            context={'request': req,
                     'ans': result,
                     'err': error_info})


if __name__ == '__main__':
    # uvicorn.run(app, host='10.0.0.4', port=80)
    uvicorn.run(app, host='172.18.143.237', port=80)
    # uvicorn.run(app, host='127.0.0.1', port=80)
    # uvicorn.run(app, host='8.209.222.49', port=80)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
