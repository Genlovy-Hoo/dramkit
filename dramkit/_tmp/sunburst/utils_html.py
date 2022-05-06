# -*- coding: utf-8 -*-

import os
import webbrowser
import bs4


def open_html(html_path, browser_path=None):
    '''
    python调用浏览器打开html文档

    Args:
        html_path: html文档路径
        browser_path: 浏览器绝对路径，若为None则以默认浏览器打开
    '''

    if browser_path is None:
        webbrowser.open(os.path.abspath(html_path)) # 直接使用默认浏览器打开
    else:
        webbrowser.register('open_html', None,
                        webbrowser.BackgroundBrowser(browser_path))
        webbrowser.get('open_html').open_new(os.path.abspath(html_path))





def html_to_soup(html_path):
    '''
     该函数用于将HTML文档读取成soup数据格式
     Input: html_path，HTML文件路径
     Output: Soup，返回Soup对象
    '''
    Fobj = open(html_path, encoding='utf-8')
    Data = Fobj.read()
    Fobj.close()
    Soup = bs4.BeautifulSoup(Data, 'lxml')
    return Soup


def soup_to_html(Soup, html_path):
    '''
    该函数用于将Soup对象写入HTML文档
    Input: Soup，需要写入的Soup对象
           html_path，HTML文件存放路径
    '''
    Fobj = open(html_path, 'w', encoding='utf-8')
    Fobj.write(str(Soup))
    Fobj.close()


def insert_js_src(html_path, js_path):
    '''
    该函数用于在HTML文档中添加一个指向js数据文件的script标签，添加位置为head标签的最后
    Input: html_path，HTML文件路径
           js_path，完整的js数据文件路径和名称
           注：若js_path为相对路径，则应以html_path的路径为参考
    '''
    # Soup对象
    Soup = html_to_soup(html_path)
    # 插入新标签
    new_tag = Soup.new_tag(name='script', src=js_path)
    Soup.head.append(new_tag)
    # 重新写入HTML文档
    soup_to_html(Soup, html_path)


def del_js_src(html_path):
    '''
    该函数用于在HTML文档中删除一个指向js数据文件的script标签，
    该标签的位置应位于head标签的末尾
    '''
    # Soup对象
    Soup = html_to_soup(html_path)
    # 删除标签
    Soup.head()[-1].decompose()
    # 重新写入HTML文档
    soup_to_html(Soup, html_path)


def get_head_num(html_path):
    '''获取head标签子标签的个数'''
    # Soup对象
    Soup = html_to_soup(html_path)
    return len(Soup.head())


def change_srcs(html_path, js_paths):
    '''用js_paths中的路径替换html中head标签里面的script标签'''
    for k in range(0, get_head_num(html_path)-len(js_paths)):
        del_js_src(html_path)
    for js_path in js_paths:
        insert_js_src(html_path, js_path)
