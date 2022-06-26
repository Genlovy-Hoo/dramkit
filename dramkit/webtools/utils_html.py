# -*- coding: utf-8 -*-

import os
import bs4
import winreg
import webbrowser
from dramkit.logtools.utils_logger import logger_show


BROWSER_REGS = {
    'ie': r'SOFTWARE\Clients\StartMenuInternet\IEXPLORE.EXE\DefaultIcon',
    'chrome': r'SOFTWARE\Clients\StartMenuInternet\Google Chrome\DefaultIcon',
    'edge': r'SOFTWARE\Clients\StartMenuInternet\Microsoft Edge\DefaultIcon',
    '360chrome': r'360ChromeURL\DefaultIcon',
    'firefox': r'SOFTWARE\Clients\StartMenuInternet\FIREFOX.EXE\DefaultIcon',
    '360': r'SOFTWARE\Clients\StartMenuInternet\360Chrome\DefaultIcon',
}    


def get_browser_path(browser='chrome', logger=None):
    '''
    | 获取浏览器的安装路径
    | param browser为浏览器名称，可选dramkit.webtools.utils_html.BROWSER_REGS中的key值
    | 参考: https://blog.csdn.net/u013314786/article/details/122497226
    '''  
    # 浏览器注册表信息
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, BROWSER_REGS[browser])
    except:
        try:
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, BROWSER_REGS[browser])
        except:
            logger_show('未找到路径，注册表信息可能错误！', logger, 'error')
            return None
    value, _type = winreg.QueryValueEx(key, '')
    return value.split(',')[0]


def open_html(path_html, browser=None, logger=None):
    '''
    | 调用浏览器打开html文档
    | path_html为html文档路径
    | browser支持浏览器名称或浏览器路径
    '''
    if browser is None:
        _browers  = ['chrome', '360chrome', 'edge', 'ie', 'firefox']
        _browers += [x for x in BROWSER_REGS.keys() if x not in _browers]
        path_browser, browser_ = None, None
        for browser in _browers:
            path_browser = get_browser_path(browser, logger=logger)
            if path_browser is not None:
                browser_ = browser
                break
        if browser_ is not None:
            webbrowser.register(browser_, None,
                                webbrowser.BackgroundBrowser(path_browser))
            webbrowser.get(browser_).open_new(os.path.abspath(path_html))
        else:
            logger_show('未正确获取浏览器路径！', logger, 'error')
    elif isinstance(browser, str) and os.path.exists(browser):
        webbrowser.register('specify', None,
                            webbrowser.BackgroundBrowser(browser))
        webbrowser.get('specify').open_new(os.path.abspath(path_html))
    else:
        path_browser = get_browser_path(browser, logger=logger)
        if path_browser is not None:
            webbrowser.register(browser, None,
                                webbrowser.BackgroundBrowser(path_browser))
            webbrowser.get(browser).open_new(os.path.abspath(path_html))
        else:
            logger_show('未正确获取{}浏览器路径！'.format(browser), logger, 'error')


def html_to_soup(path_html, encoding=None):
    '''
    | 将HTML文档读取成BeautifulSoup数据格式
    | path_html为HTML文件路径
    | 返回Soup对象
    '''
    with open(path_html, encoding=encoding) as fobj:
        data = fobj.read()
    Soup = bs4.BeautifulSoup(data, 'lxml')
    return Soup


def soup_to_html(Soup, path_html, encoding=None):
    '''
    | 将BeautifulSoup对象写入HTML文档
    | Soup为需要写入的BeautifulSoup对象
    | path_html为HTML文件存放路径
    '''
    with open(path_html, 'w', encoding=encoding) as fobj:
        fobj.write(str(Soup))


def insert_src_head_end(path_html, path_src, encoding=None):
    '''
    | 在HTML文档中添加一个引用脚本的script标签，添加位置为head标签的最后
    | path_html为HTML文件路径
    | path_src为引用的js数据文件路径和名称
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
    path_html = './_test/gauge_test.html'
    open_html(path_html)

    bsobj = html_to_soup(path_html, encoding='utf-8')
    path_html_ = './_test/gauge_text_rewright.html'
    soup_to_html(bsobj, path_html_, encoding='utf-8')
    open_html(path_html_)
    insert_src_head_end(path_html_,
                        '../../../assets/Echarts/echart_js/echarts.js',
                        encoding='utf-8')
    insert_src_head_end(path_html_,
                        '../../../assets/Echarts/echart_js/echarts.js',
                        encoding='utf-8')
    del_js_head_end(path_html_, encoding='utf-8')
