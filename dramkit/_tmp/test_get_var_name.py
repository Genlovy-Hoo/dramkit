# -*- coding: utf-8 -*-

# https://www.zhihu.com/question/42768955
# https://blog.csdn.net/Yeoman92/article/details/75076166

if __name__ == '__main__':
    a = 1
    print(dict(a=a).keys())

    def b(a):
        print(list(dict(a=a).keys())[0])
    c = 5
    b(c)


    aaa = '23asa'
    bbb = 'kjljl2'


    def get_variable_name(var):
        loc = locals()
        # print(loc)
        for key in loc:
            if loc[key] == var:
                return key


    v = get_variable_name(6)
    print(v)
    v = get_variable_name(aaa)
    print(v)
