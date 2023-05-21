# -*- coding: utf-8 -*-

# 启动方式：
# 1. python main.py
# 2. uvicorn main:app --reload

import os
import time
import uvicorn
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from dramkit.gentools import isnull
from dramkit.iotools import make_dir
from dramkit.openai.openai_chat import chat, OpenAIChat

app = FastAPI()

WAIT_SECONDS = 60*10
global CONVERS
CONVERS = {} # {ip: {'last_tm': tm, 'chater': OpenAIChat()}}

# CONVERS_DIR = None
CONVERS_DIR = './conversations/'
if not isnull(CONVERS_DIR):
    make_dir(CONVERS_DIR)

FT_DEFAULT = 'davinci:ft-personal:hsgpt-2023-03-15-10-10-47'
FT_STOP = '\n'


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
        
        
# gptchat聊天接口，单句聊天
@app.get('/gptchat0')
def gptchat0(request: Request, 
             prompt: str,
             api_key: Optional[str] = None,
             ft: Optional[str] = None,
             stop: Optional[str] = None):
    api_key = _check_api_key(api_key)
    ft = _check_ft(ft)
    stop = _check_stop(stop)
    save_path = None
    if not isnull(CONVERS_DIR):
        # TODO: 在linux服务器上这样取ip貌似时正确的，
        # 但是在windows服务上取到的ip是127.0.0.1，待解决
        ip = request.client.host
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    if not ft:
        suscess, result = chat(prompt, api_key=api_key,
                               save_path=save_path)
    else:
        suscess, result = chat(prompt, api_key=api_key,
                               save_path=save_path,
                               version='gpt3',
                               model=ft,
                               max_tokens=800,
                               stop=stop)
    res = {'result': result, 'suscess': suscess}
    return JSONResponse(content={'data': res})


# gptchat聊天接口，连续聊天
@app.get('/gptchat')
def gptchat(request: Request,
            prompt: str,
            reset: Optional[bool] = False,
            api_key: Optional[str] = None,
            ft: Optional[str] = None,
            stop: Optional[str] = None):
    api_key = _check_api_key(api_key)
    ft = _check_ft(ft)
    stop = _check_stop(stop)
    ip = request.client.host
    tmnow = time.time()
    global CONVERS
    if ip not in CONVERS or tmnow-CONVERS[ip]['last_tm'] > WAIT_SECONDS or reset:
        CONVERS[ip] = {'last_tm': tmnow, 'chater': OpenAIChat()}
    CONVERS[ip]['last_tm'] = time.time()
    CONVERS = clean_convers(CONVERS)
    print('The remaining IPs with conversation:', CONVERS.keys())
    save_path = None
    if not isnull(CONVERS_DIR):
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    if not ft:
        result, error_info = CONVERS[ip]['chater'].chat(prompt, api_key=api_key,
                                                        save_path=save_path)
    else:
        result, error_info = CONVERS[ip]['chater'].chat(
                                prompt, api_key=api_key, save_path=save_path,
                                version='gpt3',
                                model=ft,
                                max_tokens=800,
                                stop=stop)
    res = {'result': result, 'error_info': error_info}
    return JSONResponse(content={'data': res})


if __name__ == '__main__':
    uvicorn.run(app, host='10.0.0.4', port=8080)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
