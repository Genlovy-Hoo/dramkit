# -*- coding: utf-8 -*-


if __name__ == '__main__':
    import base64
    from Crypto.Cipher import AES
    from binascii import b2a_hex, a2b_hex

    AES_KEY = b'eea42cc6845811e99dcc005056895732'

    def add_to_16(text):
        if len(text.encode('utf-8')) % 16:
            add = 16 - (len(text.encode('utf-8')) % 16)
        else:
            add = 0
        text = text + ('\0' * add)
        return text.encode('utf-8')


    def encrypt(text, key=None, rtype=None):
        if key is None:
            key = AES_KEY
        else:
            key = encrypt(str(key))
        mode = AES.MODE_ECB
        text = add_to_16(text)
        print(text)
        cryptor = AES.new(key, mode)
        # text = text.encode('utf-8')
        # length = 16
        # count = len(text)
        # add = length - (count % length)
        # text = text + (b'\0' * add)

        context = cryptor.encrypt(text)
        context = b2a_hex(context)
        # return b2a_hex(context)
        # 因为AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串
        if rtype == 'base64':
            return str(base64.b64encode(context)).lstrip('b')
        else:
            return b2a_hex(context).decode('ASCII')


    # 解密后，去掉补足的空格用strip() 去掉
    def decrypt(text, key=None):
        if key is None:
            key = AES_KEY
        mode = AES.MODE_ECB
        cryptor = AES.new(key, mode)
        plain_text = cryptor.decrypt(a2b_hex(text))
        return bytes.decode(plain_text).rstrip('\0')


    text = 'hello aes'
    key = '123'
    e = encrypt(text, key=key) # 加密

    print('原始:', text)
    print('加密:', e)
    d = decrypt(e, key=encrypt(key)) # 解密
    print('解密:', d)
