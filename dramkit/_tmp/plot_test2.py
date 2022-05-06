# -*- coding: utf-8 -*-

if __name__ == '__main__':
    # https://blog.csdn.net/fengxueniu/article/details/78220392
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.linspace(-3, 3, 50) #X axis data
    y1 = np.sin(x)*x**2 +1
    
    plt.figure()
    plt.plot(x, y1)
    plt.scatter(x, y1, c='r')# set color
    
    plt.xlim((-2,2))
    plt.ylim((-2,5))
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    
    plt.show()
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.linspace(-3, 3, 50) #X axis data
    y1 = np.sin(x)*x**2 +1
    
    plt.figure()
    plt.plot(x, y1)
    plt.scatter(x, y1, c='r')# set color
    
    plt.xlim((-2,2))
    plt.ylim((-2,5))
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    
    new_ticks = np.linspace(-2,2,4)
    plt.xticks(new_ticks)
    
    y_ticks=[[-1,   1,  4],
             ['$Bad$','Normal', '$Good$']]
    plt.yticks(y_ticks[0], y_ticks[1])
    
    
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('red')
    ax.spines['bottom'].set_color('red')
    
    #xaxis.set_ticks_position设置刻度及数字名称的显示位置
    ax.xaxis.set_ticks_position('top')#set tick position para:top, bottom, both, default, none
    ax.yaxis.set_ticks_position('right')
    #使用spines设置边框，set_position设置边框位置
    ax.spines['left'].set_position(('data', 0))
    
    
    
    plt.show()
