# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import numpy as np
    from scipy import signal 
    
    from matplotlib import pyplot as plt
    
    # https://blog.csdn.net/itnerd/article/details/108362607
    # 二次曲线叠加正弦余弦-------------------------------------------------------
    N = 200
    t = np.linspace(0, 1, N)
    y = 6*t*t + np.cos(10*2*np.pi*t*t) + np.sin(6*2*np.pi*t)
    
    peaks, _ = signal.find_peaks(y, distance=1) #distance表示极大值点两两之间的距离至少大于等于5个水平单位
    
    print(peaks)
    print(len(peaks))  # the number of peaks
    
    plt.figure(figsize=(10,5))
    plt.plot(y)
    for i in range(len(peaks)):
        plt.plot(peaks[i], y[peaks[i]],'*',markersize=10)
    plt.show()
    
    # https://blog.csdn.net/chehec2010/article/details/117336967
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram
    from scipy.signal import find_peaks
    import numpy as np
    x = electrocardiogram()[2000:4000]
    peaks, _ = find_peaks(x, height=0)
    plt.figure(figsize=(10,8))
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()
    
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram
    from scipy.signal import find_peaks
    import numpy as np
    #选择大于0的
    # x = electrocardiogram()[2000:4000]
    # peaks, _ = find_peaks(x, height=0)
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    # plt.show()
     
    #小于0一下的
    x = electrocardiogram()[2000:4000]
    border = np.sin(np.linspace(0, 3 * np.pi, x.size))
    peaks, _ = find_peaks(x, height=(-border, border))
    plt.figure(figsize=(10,8))
    plt.plot(x)
    plt.plot(-border, "--", color="gray")
    plt.plot(border, ":", color="gray")
    plt.plot(peaks, x[peaks], "x")
    plt.show()
    
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram
    from scipy.signal import find_peaks
    import numpy as np
     
    #我们可以通过要求至少150个样本的距离来轻松选择心电图(ECG)中QRS络合物的位置
    x = electrocardiogram()[2000:4000]
    peaks, _ = find_peaks(x, distance=150)
    plt.figure(figsize=(10,8))
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.show()
    
    
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram
    from scipy.signal import find_peaks
    import numpy as np
     
    #通过将允许的突出限制为0.6
    x = electrocardiogram()[2000:4000]
    peaks, properties = find_peaks(x, prominence=(None, 0.6))
    plt.figure(figsize=(10,8))
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.show()
    
    
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram
    from scipy.signal import find_peaks
    import numpy as np
     
    #要仅选择非典型心跳，我们将两个条件结合起来：最小突出1和至少20个样本的宽度。
    x = electrocardiogram()[17000:18000]
    peaks, properties = find_peaks(x, prominence=1, width=20)
    
    plt.figure(figsize=(10,8))
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
               ymax = x[peaks], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
              xmax=properties["right_ips"], color = "C1")
    plt.show()
    
