# -*- coding: utf-8 -*-

import easyocr

if __name__ == '__main__':
    #设置识别中英文两种语言
    reader = easyocr.Reader(['ch_sim','en'], gpu = False) # need to run only once to load model into memory
    result = reader.readtext("微信截图_20220114095716.png", detail = 0)
    print(result)
