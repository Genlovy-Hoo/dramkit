# -*- coding: utf-8 -*-

if __name__ == '__main__':
    from dramkit.wechat import WechatWork
    
    corpid = 'wwff25dd15181d772d' # 企业id
    appid = 1000002 # 应用id
    # 应用token
    corpsecret = 'w7kYZr4maLGDJ3Na_IdRhtss2znQtUJviFya7Qj6Ses'
    
    users = ['somebody'] # 接收人
    
    w = WechatWork(corpid=corpid,
                   appid=appid,
                   corpsecret=corpsecret)
    
    # 发送文本
    res = w.send_text('test!', users)
    
    # # 发送文件
    # res = w.send_file('./GDP2015.csv', users)
    
    # # 发送图片
    # res = w.send_image('./吉祥三宝.jpg', users)
    
    # # 发送 Markdown
    # res = w.send_markdown('# Hello World', users)
    
    
