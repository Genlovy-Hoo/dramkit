# -*- coding: utf-8 -*-

# https://blog.csdn.net/rhx_qiuzhi/article/details/124332114

# deom4
import time
import asyncio

async def washing1():
    await asyncio.sleep(3)
    print('小朋友的衣服洗完了')
    return 1

async def washing2():
    await asyncio.sleep(2)
    print('爷爷奶奶的衣服洗完了')
    return 2


async def washing3():
    await asyncio.sleep(5)
    print('爸爸妈妈的衣服洗完了')
    return 3


if __name__ == '__main__':
    # 1. 创建一个事件循环
    loop = asyncio.get_event_loop()
    startTime = time.time()
    # 2. 将异步函数加入事件队列
    tasks = [
        washing1(),
        washing2(),
        washing3(),
    ]
    # 3.执行队列实践，直到最晚的一个事件被处理完毕后结束
    loop.run_until_complete(asyncio.wait(tasks))
    # 4.如果不在使用loop，建议使用关闭，类似操作文件的close()函数
    loop.close()
    endTime = time.time()
    print("洗完三批衣服共耗时: ",endTime-startTime)

