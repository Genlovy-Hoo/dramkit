# -*- coding: utf-8 -*-

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


def encrypt(text, key=None):
    if key is None:
        key = AES_KEY
    else:
        key = encrypt(str(key))
    mode = AES.MODE_ECB
    text = add_to_16(text)
    cryptos = AES.new(key, mode)

    cipher_text = cryptos.encrypt(text)
    return b2a_hex(cipher_text)


def decrypt(text, key=None):
    if key is None:
        key = AES_KEY
    mode = AES.MODE_ECB
    cryptor = AES.new(key, mode)
    plain_text = cryptor.decrypt(a2b_hex(text))
    return bytes.decode(plain_text).rstrip('\0')


if __name__ == '__main__':
    text = '原始字符串'
    key = '这是key'
    e = encrypt(text, key=key) # 加密
    d = decrypt(e, key=encrypt(key)) # 解密
    print('原始:', text)
    print('加密:', e)
    print(e.decode('utf-8'))
    print('解密:', d)
