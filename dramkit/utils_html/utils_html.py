# -*- coding: utf-8 -*-

import os
import bs4
import webbrowser
from utils_hoo.utils_general import isnull


def open_html(path_html, path_chrome=None):
    '''调用浏览器打开html文档'''
    if isnull(path_chrome):
        paths = [
            'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
            'C:/Program Files/Google/Chrome/Application/chrome.exe',
            'C:/Users/EDZ/AppData/Local/360Chrome/Chrome/Application/360chrome.exe'
            'C:/Users/glhyy/AppData/Local/360Chrome/Chrome/Application/360chrome.exe',
            ]
        for path in paths:
            if os.path.exists(path):
                path_chrome = path
                break
    webbrowser.register('chrome', None,
                        webbrowser.BackgroundBrowser(path_chrome))
    webbrowser.get('chrome').open_new(os.path.abspath(path_html))
    
    
def html_to_soup(path_html, encoding=None):
    '''
    将HTML文档读取成BeautifulSoup数据格式
    path_html为HTML文件路径
    返回Soup对象
    '''
    with open(path_html, encoding=encoding) as fobj:
        data = fobj.read()
    Soup = bs4.BeautifulSoup(data, 'lxml')
    return Soup

    
def soup_to_html(Soup, path_html, encoding=None):
    '''
    将BeautifulSoup对象写入HTML文档
    Soup为需要写入的BeautifulSoup对象
    path_html为HTML文件存放路径
    '''
    with open(path_html, 'w', encoding=encoding) as fobj:
        fobj.write(str(Soup))
        
        
def insert_src_head_end(path_html, path_src, encoding=None):
    '''
    在HTML文档中添加一个引用脚本的script标签，添加位置为head标签的最后
    path_html为HTML文件路径
    path_src为引用的js数据文件路径和名称
    '''    
    Soup = html_to_soup(path_html, encoding=encoding) # 读取为BeautifulSoup对象
    new_tag = Soup.new_tag(name='script', src=path_src) 
    Soup.head.append(new_tag) # 插入新标签    
    soup_to_html(Soup, path_html, encoding=encoding) # 重新写入HTML文档
        

def del_js_head_end(path_html, encoding=None):
    '''
    删除HTML文档head部分的最后一个标签
    '''
    Soup = html_to_soup(path_html, encoding=encoding) # 读取为BeautifulSoup对象
    Soup.head()[-1].decompose() # 删除标签    
    soup_to_html(Soup, path_html, encoding=encoding) # 重新写入HTML文档
    
    
def get_head_num(path_html, encoding=None):
    '''获取head标签子标签的个数'''
    Soup = html_to_soup(path_html, encoding=encoding) # 读取为BeautifulSoup对象
    return len(Soup.head())
    
    
if __name__ == '__main__':
    path_html = './test/gauge_test.html'
    open_html(path_html)
    
    bsobj = html_to_soup(path_html, encoding='utf-8')
    path_html_ = './test/gauge_text_rewright.html'
    soup_to_html(bsobj, path_html_, encoding='utf-8')
    open_html(path_html_)
    insert_src_head_end(path_html_,
                        '../../utils_plot/Echarts/echart_js/echarts.js',
                        encoding='utf-8')
    del_js_head_end(path_html_, encoding='utf-8')
    
    
    
    
    
    
    
    
    
    
    