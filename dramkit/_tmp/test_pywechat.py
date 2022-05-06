# -*- coding: utf-8 -*-

"""
实现思路
1、如何输入文字的？        输入法 -- 控制键盘消息 -- 文字输入
2、如何判断输入框在哪里？   通过鼠标左键点击 -- 确定输入框的位置 -- 确保消息正确发送
3、怎么发消失文字？        通过模拟回车键实现发送消息
4、如何一直发送消息？      使用循环的方式实现
"""
 
# 第三方的库 pip install pynput
from pynput.keyboard import Key, Controller as key_cl  # 键盘控制器
from pynput.mouse import Button, Controller as mouse_cl  # 鼠标的控制器
import time  # 引入时间
import emoji  # 引入表情 pip install emoji
 
 
# 键盘的控制函数
def keyboard_input(msg):
    keyboard = key_cl()  # 使用管理员来获取键盘的权限
    keyboard.type(msg)  # 设置发送数据的类型
 
 
# 鼠标的控制函数
def mouse_click():
    mouse = mouse_cl()  # 获取鼠标管理员权限
    mouse.press(Button.left)  # 模拟鼠标左键的按下
    mouse.release(Button.left)  # 模拟鼠标左键的弹起
 
 
# 实现消息的发送函数
# num:发送的次数 msg:发送的消息
def send_message(num, msg):
    print("程序在五秒后开始执行，预留一点操作时间")
    time.sleep(5)
    keyboard = key_cl()
 
    for i in range(num):
        keyboard_input(msg)
        mouse_click()
        time.sleep(1)              # 消息间隔延迟时间，不可太快，容易被微信拦截
        keyboard.press(Key.enter)    # 模拟回车键的按下
        keyboard.release(Key.enter)  # 模拟回车键的弹起
 
 
if __name__ == '__main__':
    # 当你惹女友生气时:发给女朋友
    send_message(3, "csdn" + emoji.emojize(':red_heart:') + emoji.emojize(':red_heart:'))