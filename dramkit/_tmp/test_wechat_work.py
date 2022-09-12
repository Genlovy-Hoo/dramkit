# -*- coding: utf-8 -*-

# https://m.zhuxianfei.com/python/47631.html
  
import requests, sys
  
  
class SendWeiXinWork():
    def __init__(self):
        self.CORP_ID= "wwff25dd15181d772d"  # 企业号的标识
        self.SECRET= "w7kYZr4maLGDJ3Na_IdRhtss2znQtUJviFya7Qj6Ses"  # 管理组凭证密钥
        self.AGENT_ID= 1000002 # 应用ID
        self.token= self.get_token()
  
    def get_token(self):
        url= "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
        data= {
            "corpid":self.CORP_ID,
            "corpsecret":self.SECRET
        }
        req= requests.get(url=url, params=data)
        res= req.json()
        if res['errmsg']== 'ok':
            return res["access_token"]
        else:
            return res
  
    def send_message(self, to_user, content):
        url= "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s" % self.token
        data= {
            "touser": to_user,  # 发送个人就填用户账号
            # "toparty": to_user, # 发送组内成员就填部门ID
            "msgtype":"text",
            "agentid":self.AGENT_ID,
            "text": {"content": content},
            "safe":"0"
        }
  
        req= requests.post(url=url, json=data)
        res= req.json()
        if res['errmsg']== 'ok':
            print("send message sucessed")
            return "send message sucessed"
        else:
            return res
  
  
if __name__== '__main__':
    SendWeiXinWork= SendWeiXinWork()
    a = SendWeiXinWork.send_message("somebody","测试a")