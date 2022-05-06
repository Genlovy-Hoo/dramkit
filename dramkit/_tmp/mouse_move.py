# -*- coding: utf-8 -*-

from ctypes import windll
import win32api
import win32con
import time


if __name__ == '__main__':
    win_width = windll.user32.GetSystemMetrics(0)
    win_height = windll.user32.GetSystemMetrics(1)

    rd = 0
    t1 = time.time()

    while True:
        rd = (rd+1) % 1000
        temp = rd % 4
        if temp == 0:
            w_move = 100
            h_move = 0
        elif temp == 1:
            w_move = 0
            h_move = 100
        elif temp == 2:
            w_move = -100
            h_move = 0
        else:
            w_move = 0
            h_move = -100

        width, height = win32api.GetCursorPos()
        width += w_move
        height += h_move
        windll.user32.SetCursorPos(width, height)
        if time.time() - t1 > 60:
            print(width, height)
            t1 = time.time()

        time.sleep(10)
