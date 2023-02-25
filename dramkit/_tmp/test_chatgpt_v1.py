# -*- coding: utf-8 -*-


if __name__ == '__main__':
    from revChatGPT.V1 import Chatbot

    # 只填邮箱和用户名必须翻墙，否则会登录失败获取不了access_token报错
    # 若使用access_token，在登录OpenAI官网之后从下面这个页面复制：
    # https://chat.openai.com/api/auth/session
    chatbot = Chatbot(config={
       "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJnZW5sb3ZoeXlAMTYzLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJnZW9pcF9jb3VudHJ5IjoiVVMifSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLVE4NzNSR2RyMTByYU43OUZjb29UOEw3WiJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjM5MTQxMmVkYTBhOWQzZDU3ZDZmMDAxIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY3NjUyNjc3MSwiZXhwIjoxNjc3NzM2MzcxLCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9mZmxpbmVfYWNjZXNzIn0.JB4EYI6HyDumT7yonDzUcxMN451chcP4PRFA4OcMpfiQoxZkzL434hXc3PEa3Uibl9YZZGZrJzQDq32sNqt4f-vKXgKy24B4sd8X_YW2f8GMZp2ZGFIdxqNuZ0Z7nrTFl4NGhacE1pIu6jnEj-lJXBa5oA9VNlAYCJPYTPDj59s7ERI0c7aNpE4c_L2ippw0syL1NTSxJQggKpoPHY2c6WiH-F0pTG1uJxtGU5CCb9RyOQ0IXkIuM946PEfrluz_vSl_ig0YVdKWqLyXxLbzDNIY6-bEDfdbaNs8DRZTrEXWbXBaKDd4s4nkgBTES90U5uhSX6JMtkmzJf2UJgljlg"
      # "email": "genlovhyy@163.com",
      # "password": "GLHYYopenai"
    })
    
    print("Chatbot: ")
    prev_text = ""
    for data in chatbot.ask(
        # "Hello world",
        "以杭州西湖为主题写一首诗："
    ):
        message = data["message"][len(prev_text) :]
        print(message, end="", flush=True)
        prev_text = data["message"]
    print()
