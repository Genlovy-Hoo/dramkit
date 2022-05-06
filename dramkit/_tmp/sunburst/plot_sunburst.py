# -*- coding: utf-8 -*-

import pandas as pd
from pyecharts.charts import Sunburst
from pyecharts import options as opts


import os
import webbrowser


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



fpath = './w20211127.xls'
data = pd.read_excel(fpath)
data.fillna(method='ffill', inplace=True)
parents = data['Term'].unique().tolist()
_data = []
for parent in parents:
    tmp = {}
    tmp['name'] = parent.split(':')[0]
    children = data[data['Term'] == parent]['Genes'].tolist()
    children = [{'name': x.strip(), 'value': 1} for x in children]
    tmp['children'] = children
    _data.append(tmp)
data = _data


c = (
    Sunburst(init_opts=opts.InitOpts(
        width="1500px", height="700px"))
    .add(
        "",
        data_pair=data,
        highlight_policy="ancestor",
        radius=[0, "95%"],
        sort_="null",
        levels=[
            {},
            {"r0": "1%", "r": "50%",
             "label": {"align": "center"}},
            {
                "r0": "50%",
                "r": "80%",
                "label": { "position": "inside", "padding": 3,
                          "silent": False, 'fontSize': 8},
                "itemStyle": {"borderWidth": 2},
            },
        ],
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Sunburst-官方示例"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))
    .render("drink_flavors.html")
)

open_html("drink_flavors.html")
