# -*- coding -*-
# 归并排序，文波

def merge(li, low, mid, high):
    i = low
    j = mid + 1
    ltmp = []
    while i <= mid and j <= high:
        if li[i] < li[j]:
            ltmp.append(li[i])
            i += 1
        else:
            ltmp.append(li[j])
            j += 1
    while i <= mid:
        ltmp.append(li[i])
        i += 1
    while j <= high:
        ltmp.append(li[j])
        j += 1
    li[low:high+1] = ltmp


def merge_sort(li, low, high):
    if low < high:
        mid = (low + high) // 2
        merge_sort(li, low, mid)
        merge_sort(li, mid+1, high)
        merge(li, low, mid, high)


if __name__ == '__main__':
    li = [2, 4, 5, 7, 1, 3, 6, 8]
    merge(li, 0, 3, 7)
    print(li)

    li = list(range(1000))
    import random
    random.shuffle(li)
    print(li)
    merge_sort(li, 0, len(li)-1)
    print(li)
