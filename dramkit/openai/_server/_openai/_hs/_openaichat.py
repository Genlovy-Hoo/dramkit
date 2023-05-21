# -*- coding: utf-8 -*-

# pip install dramkit -U

from dramkit.openai.openai_chat import OpenAIChat
chater = OpenAIChat()
res = chater.chat('中国有多少个地级市')
print(res)
