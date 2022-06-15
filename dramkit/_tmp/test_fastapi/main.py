# -*- coding: utf-8 -*-

import uvicorn
from fastapi import FastAPI


app=FastAPI()


# 添加首页
@app.get('/')
def index():
    '''首页'''
    return 'This is Home Page.'


@app.get('/user')
def user():
    '''用户'''
    return {'msg': 'Get all users', 'code': 2001}


@app.post('/login')
def login():
    '''登录'''
    return {'msg': 'login suscess'}


@app.api_route('/manymethods', methods=['GET', 'POST', 'PUT'])
def manymethods():
    return {'msg': 'This if hen duo methods'}


if __name__ == '__main__':
    uvicorn.run(app)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    