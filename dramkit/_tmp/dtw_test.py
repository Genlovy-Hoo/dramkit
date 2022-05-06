# -*- coding: utf-8 -*-

# https://dynamictimewarping.github.io/python/
# https://pypi.org/project/dtw-python/

if __name__ == '__main__':
    import numpy as np

    ## A noisy sine wave as query
    idx = np.linspace(0,6.28,num=100)
    query = np.sin(idx) + np.random.uniform(size=100)/10.0

    ## A cosine is for template; sin and cos are offset by 25 samples
    template = np.cos(idx)

    ## Find the best match with the canonical recursion formula
    from dtw import dtw, rabinerJuangStepPattern
    alignment = dtw(query, template, keep_internals=True)

    ## Display the warping curve, i.e. the alignment curve
    alignment.plot(type="threeway")

    ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    dtw(query, template, keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c"))\
        .plot(type="twoway",offset=-2)

    ## See the recursion relation, as formula and diagram
    print(rabinerJuangStepPattern(6,"c"))
    rabinerJuangStepPattern(6,"c").plot()


    # import numpy as np

    # # We define two sequences x, y as numpy array
    # # where y is actually a sub-sequence from x
    # x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    # y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

    # from dtw import dtw

    # manhattan_distance = lambda x, y: np.abs(x - y)

    # d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

    # print(d)

    # # You can also visualise the accumulated cost and the shortest path
    # import matplotlib.pyplot as plt

    # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.show()
