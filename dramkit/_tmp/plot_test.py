# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpathes

    fig,ax = plt.subplots()
    xy1 = np.array([0.2,0.2])
    xy2 = np.array([0.2,0.8])
    xy3 = np.array([0.8,0.2])
    xy4 = np.array([0.8,0.8])
    xy5 = np.array([0.45,0.5])
    #圆形
    circle = mpathes.Circle(xy1,0.05)
    ax.add_patch(circle)
    #三角形
    rect = mpathes.RegularPolygon(xy5,3,0.1,color='r')
    ax.add_patch(rect)
    #长方形
    rect = mpathes.Rectangle(xy2,0.2,0.1,color='r')
    ax.add_patch(rect)
    #多边形
    polygon = mpathes.RegularPolygon(xy3,5,0.1,color='g')
    ax.add_patch(polygon)
    #椭圆形
    ellipse = mpathes.Ellipse(xy4,0.4,0.2,color='y')
    ax.add_patch(ellipse)

    plt.axis('equal')
    plt.grid()
    plt.show()


    import matplotlib.pyplot as plt

    # 位置
    l = [[120.7015202,36.37423,0],
    [120.7056165,36.37248342,4],
    [120.70691,36.37579616,3],
    [120.7031731,36.37753964,5],
    [120.7011609,36.37905063,10],
    [120.6973521,36.37876006,8],
    [120.6928965,36.37800457,6],
    [120.6943337,36.37521499,7],
    [120.6962022,36.37643544,9],
    [120.6987175,36.37457569,1],
    [120.6997954,36.37591239,2],
    [120.7015202,36.37423,0]]


    def drawPic(dots):
        plt.figure(figsize=(10,6))
        plt.xlim(120.692,120.708,0.002)
        plt.ylim(36.372,36.380,0.001)
        # for i in range(len(dots)-1):
        #     plt.text(l[i][0],l[i][1],'C'+str(l[i][2]),color='#0085c3',fontproperties="simhei")
        #     plt.plot(l[i][0],l[i][1],'o',color='#0085c3')
        #连接各个点
        for i in range(len(dots)-1):
            print(i)
            start = (l[i][0],l[i+1][0])
            end = (l[i][1],l[i+1][1])
            plt.plot(start,end,'-o',color='#0085c3')
        plt.show()

    drawPic(l)

    # x轴放在顶部
    values = [-1, -2, -3, -1, -5, -4]
    fig, ax = plt.subplots()
    ax.plot(values, '.-r')
    # ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.show()


    plt.figure()
    ax = plt.subplot(111)
    ax.plot([1,2,3,1,5,6])
    ax.fill_between([1, 3], [1, 5], color='grey', alpha=0.2)
    ax.fill_betweenx([2, 4], [1, 5], color='blue', alpha=0.5)
    ax.fill_between([3.5, 5], 0, 6, color='red', alpha=0.5)
    print(ax.axis())
    ax.fill_between([3.5, 5], ax.axis()[2], ax.axis()[3],
                    color='yellow', alpha=0.5)
    ax.fill_betweenx([5, 6], ax.axis()[0], ax.axis()[1],
                     color='black', alpha=0.5)
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    ax2 = ax.twinx()
    y2 = [0.0, -0.1, -0.2, -0.25, -0.2, -0.05, -0.1]
    # ax2.stackplot(range(0, len(y2)), y2, color='c', alpha=0.3)
    ax2.plot(y2, '-', alpha=0.3)
    ax2.fill_between(range(0, len(y2)), y2, color='c', alpha=0.3)
    ax2.set_ylim([-0.5, 0.0])
    # ax2.invert_xaxis()
    y1 = [1,2,3,1,5,6]
    ax.plot(y1, '.-k')
    ax.axhline(y=2.5, c='r', ls='-', lw=1.5, alpha=1)
    # ax.stackplot(range(0, len(y1)), y1)
    ax2.spines['top'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax2.spines['right'].set_color('none')
    # ax.spines['right'].set_color('none')
    ax2.xaxis.set_ticks_position('top')
    ax.xaxis.set_ticks_position('top')
    plt.show()



    import numpy as np
    import matplotlib.pyplot as plt

    a = [5, 10, 15, 10, 5]
    b = [5, 5, 10, 15, 10]

    c = [5, 5, 10, 15, 10]
    d = [5, 10, 5, 10, 15]

    e = [1, 2, 2, 1]
    f = [3, 3, 4, 4]

    plt.subplot(221)
    plt.plot(a, b, '-o')
    plt.fill(a, b, 'r')
    for index, item in enumerate(zip(a, b), 1):
        plt.text(item[0], item[1], index)

    plt.subplot(222)
    plt.plot(c, d, 'o')
    plt.fill(c, d, alpha=0.5)
    for index, item in enumerate(zip(c, d), 1):
        plt.text(item[0], item[1], index)

    plt.subplot(223)
    plt.fill(e, f)

    plt.subplot(224)
    plt.fill("time", "signal",
             data={"time": [2, 4, 4], "signal": [3, 4, 3]})

    plt.show()
