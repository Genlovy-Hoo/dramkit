# -*- coding: utf-8 -*-


def twinx_align(ax_left, ax_right, v_left, v_right):
    '''
    实现左轴(ax_left)在v_left位置和右轴(ax_right)在v_right位置对齐
    '''
    left_min, left_max = ax_left.get_ybound()
    right_min, right_max = ax_right.get_ybound()
    k = (left_max-left_min) / (right_max-right_min)
    b = left_min - k * right_min
    x_right_new = k * v_right + b
    dif = x_right_new - v_left
    # 第一次线性映射，目的是获取左轴和右轴的伸缩比例，通过这层映射，
    # 计算可以得到右轴指定位置映射到左轴之后在左轴的位置。
    # 右轴目标映射到左轴的位置与左轴目标位置存在一个差值，
    # 这个差值就是右轴的一端需要扩展的距离，这个距离是映射到左轴之后按左轴的尺度度量的。
    # 通过第一次线性映射的逆映射，计算得到右轴一端实际需要扩展的距离。
    # 得到右轴一端的扩展距离之后，右轴就有两个固定点：一个端点和一个目标点。
    # 将右轴这两个固定点与左轴对应的点做第二次线性映射，可以再次得到两轴的伸缩比例，
    # 得到新的伸缩比例之后，通过左轴的另一个端点进行逆映射，可以计算题右轴的另一个端点。
    # 最后通过右轴两个端点位置以及新的伸缩比例对右轴进行伸缩变化，
    # 即可将左轴与右轴在指定刻度位置对齐。
    if dif >= 0:
        right_min_new = ((left_min-dif) - b) / k
        k_new = (left_min-v_left) / (right_min_new-v_right)
        b_new = v_left - k_new * v_right
        right_max_new = (left_max - b_new) / k_new
    else:
        right_max_new = ((left_max-dif) - b) / k
        k_new = (left_max-v_left) / (right_max_new-v_right)
        b_new = v_left - k_new * v_right
        right_min_new = (left_min - b_new) / k_new    
    def _forward(x):
        return k_new * x + b_new
    def _inverse(x):
        return (x - b_new) / k_new
    ax_right.set_ylim([right_min_new, right_max_new])
    ax_right.set_yscale('function', functions=(_forward, _inverse))
    return ax_left, ax_right






