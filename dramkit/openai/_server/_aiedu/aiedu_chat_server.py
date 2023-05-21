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
from pydantic import BaseModel

from dramkit.gentools import isnull
from dramkit.iotools import make_dir
from dramkit.openai.aiedu_chat import chat, OpenAIChat

app = FastAPI()

WAIT_SECONDS = 60*10
global CONVERS
CONVERS = {} # {ip: {'last_tm': tm, 'chater': OpenAIChat()}}

# CONVERS_DIR = None
CONVERS_DIR = './conversations/'
if not isnull(CONVERS_DIR):
    make_dir(CONVERS_DIR)


def clean_convers(CONVERS):
    res = {}
    for ip in CONVERS:
        if time.time()-CONVERS[ip]['last_tm'] <= WAIT_SECONDS:
            res[ip] = CONVERS[ip]
    return res


class OpenaiParams(BaseModel):
    question: str
    model: str = 'gpt-3.5-turbo'
        
        
# gptchat聊天接口，单句聊天
@app.post('/gptchat0')
async def gptchat0(request: Request, 
                   params: Optional[OpenaiParams]
                   ):
    params_ = params.dict().copy()
    prompt = params_.pop('question')
    save_path = None
    if not isnull(CONVERS_DIR):
        ip = request.client.host
        save_path = os.path.join(CONVERS_DIR, '%s.json'%ip)
    suscess, result = chat(prompt, save_path=save_path,
                           **params_)
    res = {'result': result, 'suscess': suscess}
    return JSONResponse(content={'data': res})


# gptchat聊天接口，连续聊天
@app.post('/gptchat')
async def gptchat(request: Request,
                  params: Optional[OpenaiParams],
                  reset: Optional[bool] = False
                  ):
    params_ = params.dict().copy()
    prompt = params_.pop('question')
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
    result, error_info = CONVERS[ip]['chater'].chat(
                         prompt, save_path=save_path,
                         **params_)
    res = {'result': result, 'error_info': error_info}
    return JSONResponse(content={'data': res})


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
