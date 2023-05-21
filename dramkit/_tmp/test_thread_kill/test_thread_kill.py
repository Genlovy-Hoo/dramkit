# -*- coding: utf-8 -*-

if __name__ == '__main__':

    # https://zhuanlan.zhihu.com/p/208260624
    
    import sys
    import threading 
    import time 
    class thread_with_trace(threading.Thread): 
        def __init__(self, *args, **keywords):
            threading.Thread.__init__(self, *args, **keywords)
            self.killed = False
            self.result = None
            
        def run(self):
            sys.settrace(self.globaltrace)
            self.result = self._target(*self._args, **self._kwargs)
        
        # def start(self): 
        #  	self.__run_backup = self.run 
        #  	self.run = self.__run	 
        #  	threading.Thread.start(self) 
        
        # def __run(self): 
        #     sys.settrace(self.globaltrace)
        #     self.result = self.__run_backup()
        #     self.run = self.__run_backup 
        
        def globaltrace(self, frame, event, arg): 
            if event == 'call': 
                return self.localtrace
            else: 
                return None
        
        def localtrace(self, frame, event, arg):
            if self.killed: 
                if event == 'line':
                    raise SystemExit()
            return self.localtrace 
        
        def kill(self): 
        	self.killed = True
        
    def func(x, **kwargs): 
        # while True: 
        print('thread running')
        with open('a.xlsx') as f:
            # import pandas as pd
            # dfe = pd.read_excel(f)
            time.sleep(x)
        print(kwargs)
        return x
        
    from dramkit import TimeRecoder
    tr = TimeRecoder()
    
    t1 = thread_with_trace(target=func,args=[6,], kwargs={'y':3}) 
    t1.start() 
    # time.sleep(2)  
    t1.join(3) 
    print(t1.result)
    t1.kill()
    # time.sleep(1)
    # print(5)
    # t1.join()
    if not t1.is_alive(): 
        print('thread killed') 
    import pandas as pd
    df = pd.read_excel('b.xlsx')
    df.to_csv('a.xlsx')
        
    tr.used()
    
    
    
    
    
    
    
    
    
    # Python program showing 
    # how to kill threads 
    # using set/reset stop 
    # flag 
 

    def run(): 
    	while True: 
    		print('thread running') 
    		global stop_threads 
    		if stop_threads: 
    			break

    stop_threads = False
    t1 = threading.Thread(target = run) 
    t1.start() 
    time.sleep(1) 
    stop_threads = True
    t1.join() 
    print('thread killed')