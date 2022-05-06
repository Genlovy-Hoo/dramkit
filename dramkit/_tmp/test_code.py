# -*- coding: utf-8 -*-

if __name__ == '__main__':

    oath = '我爱妞'
    print(type(oath))
    print(len(oath))

    oath1 = u'我爱妞'
    print(type(oath1))
    print(len(oath1))

    print(oath==oath1)


    utf8 = oath.encode('utf-8')
    print(type(utf8))
    print(len(utf8))
    print(utf8)

    gbk = oath.encode('gbk')
    print(type(gbk))
    print(len(gbk))
    print(gbk)


    out = open('test.txt','w',encoding = 'utf-8')

    test = u'\u5220\u9664'
    print(len(test))
    print(test)
    test1 = test.encode('utf-8')
    print(test1)
    print(type(test1))

    out.write(test)
    out.close()
