# -*- coding: utf-8 -*-

import time
import asyncio


def sleep(x=1, t=5):
    print(locals())
    print('sleeping')
    time.sleep(t)
    return x+t


async def sleep(x=1, t=5):
    print(locals())
    print('sleeping')
    await asyncio.sleep(t)
    return x+t


async def get_input():
    return input('please input:')


def await_func_timeout(func, timeout=5, *args, **kwargs):
    print(locals())
    
    async def _await_run():        
        try:
            # return await asyncio.wait_for(_await_func(),
            #                               timeout=timeout)
            return await asyncio.wait_for(func(*args, **kwargs),
                                          timeout=timeout)
        except asyncio.TimeoutError:
            print('任务超时，已取消！')
            
    return asyncio.run(_await_run())

    
a = await_func_timeout(sleep, timeout=6)
print(a)
    
# # SuperFastPython.com
# # example of waiting for a coroutine with a timeout
# from random import random-
# # coroutine to execute in a new task
# async def task_coro(arg):
#     # generate a random value between 0 and 1
#     value = 1 + random()
#     # report message
#     print(f'>task got {value}')
#     # block for a moment
#     await asyncio.sleep(value)
#     # report all done
#     print('>task done')
# # main coroutine
# async def main():
#     # create a task
#     task = task_coro(1)
#     # execute and wait for the task without a timeout
#     try:
#         await asyncio.wait_for(task, timeout=0.2)
#     except asyncio.TimeoutError:
#         print('Gave up waiting, task canceled')
# # start the asyncio program
# asyncio.run(main())