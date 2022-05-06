# -*- coding: utf-8 -*-

import numpy as np

def vmd( signal, alpha, tau, K, DC, init, tol):
    '''
    用VMD分解算法时只要把信号输入进行分解就行了，只是对信号进行分解，和采样频率没有关系，
    VMD的输入参数也没有采样频率。
    VMD分解出的各分量在输出量 u 中，这个和信号的长度、信号的采样频率没有关系。
    迭代时各分量的中心频率在输出量omega，可以用2*pi/fs*omega求出中心频率，
    但迭代时的频率是变化的。

    Input and Parameters:
    signal  - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                       2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    '''

    # Period and sampling frequency of input signal
    #分解算法中的采样频率和时间是标准化的，分解信号的采样时间为1s,然后就得到相应的采样频率。采样时间间隔：1/ length(signal)，频率： length(signal)。
    save_T = len(signal)
    fs = 1 / save_T
    # extend the signal by mirroring镜像延拓
    T = save_T
    f_mirror = []
    temp = signal[0:T//2]
    f_mirror.extend(temp[::-1]) #temp[::-1] 倒序排列
    f_mirror.extend(signal)
    temp = signal[T//2:T]
    f_mirror.extend(temp[::-1])

    f = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = [(i + 1) / T for i in range(T)]  # 列表从1开始
    # Spectral Domain discretization
    #freqs 进行移位是由于进行傅里叶变换时，会有正负对称的频率，分析时一般只有正频率，所以看到的频谱图是没有负频率的
    freqs = np.array( [i - 0.5 - 1 / T for i in t] )

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)
    # Construct and center f_hat
    transformed = np.fft.fft(f)  # 使用fft函数对信号进行快速傅里叶变换。
    f_hat = np.fft.fftshift(transformed)  # 使用fftshift函数进行移频操作。
    f_hat_plus = f_hat
    f_hat_plus[0:T // 2] = 0
    # f_hat_plus[0:T] = 1                #????????????????????????????////////////

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = [np.zeros((N, len(freqs)), dtype=complex) for i in range(K)]
    # Initialization of omega_k
    omega_plus = np.zeros((N, K))

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))
    else:
        omega_plus[0, :] = 0
        # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0
        # start with empty dual variables
    lambda_hat = np.zeros( (N, len(freqs)), dtype=complex)
    # other inits
    eps = 2.2204e-16  # python里没有eps功能
    uDiff = tol + eps  # update step
    n = 1  # loop counter
    sum_uk = 0  # accumulator


    #----------- Main loop for iterative updates----------
    while (uDiff > tol and  n < N ):    #not converged and below iterations limit
        #update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[K-1][n-1,:]+ sum_uk - u_hat_plus[0][n-1,:]  #sum_uk 一直都等于0（1,2000）????????????????
        #update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[k][n,:] = (f_hat_plus - sum_uk - lambda_hat[n-1,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n-1,k])**2)
        #update first omega if not held at 0
        if not DC:
            omega_plus[n,k] = (freqs[T//2:T]*np.mat(np.abs(u_hat_plus[k][n, T//2:T])**2).H)/np.sum(np.abs(u_hat_plus[k][n,T//2:T])**2)

        #update of any other mode
        for k in range(K-1):
            #accumulator
            sum_uk = u_hat_plus[k][n,:] + sum_uk - u_hat_plus[k+1][n-1,:]
            #mode spectrum
            u_hat_plus[k+1][n,:] = (f_hat_plus - sum_uk - lambda_hat[n-1,:]/2)/(1+Alpha[k+1]*(freqs - omega_plus[n-1,k+1])**2)
            #center frequencies
            omega_plus[n,k+1] = (freqs[T//2:T]*np.mat(np.abs(u_hat_plus[k+1][n, T//2:T])**2).H)/np.sum(np.abs(u_hat_plus[k+1][n,T//2:T])**2)

        #Dual ascent
        lambda_hat[n,:] = lambda_hat[n-1,:] + tau*(np.sum([ u_hat_plus[i][n,:] for i in range(K)],0) - f_hat_plus)
        #loop counter
        n = n+1
        #converged yet?
        uDiff = eps
        for i in range(K):
            uDiff = uDiff + 1/T*(u_hat_plus[i][n-1,:]-u_hat_plus[i][n-2,:])*np.mat((u_hat_plus[i][n-1,:]-u_hat_plus[i][n-2,:]).conjugate()).H
        uDiff = np.abs(uDiff)


    # ------ Postprocessing and cleanup-------

    #discard empty space if converged early
    N = min(N,n)
    omega = omega_plus[0:N,:]
    #Signal reconstruction
    u_hat = np.zeros((T, K), dtype=complex)
    temp = [u_hat_plus[i][N-1,T//2:T] for i in range(K) ]
    u_hat[T//2:T,:] = np.squeeze(temp).T

    temp = np.squeeze(np.mat(temp).conjugate())
    u_hat[1:(T//2+1),:] = temp.T[::-1]

    u_hat[0,:] = (u_hat[-1,:]).conjugate()

    u = np.zeros((K,len(t)))

    for k in range(K):
        u[k,:]=np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
    #remove mirror part
    u = u[:,T//4:3*T//4]
    #recompute spectrum
    u_hat = np.zeros((T//2, K), dtype=complex)
    for k in range(K):
        u_hat[:,k]= np.squeeze( np.mat( np.fft.fftshift(np.fft.fft(u[k,:])) ).H)

    return u, u_hat, omega
