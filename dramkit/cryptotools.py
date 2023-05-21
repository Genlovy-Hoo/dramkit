# -*- coding: utf-8 -*-

# 参考：
# http://www.noobyard.com/article/p-dkyisoqc-gv.html
# https://www.cnblogs.com/testlearn/p/16177034.html
# https://www.cnblogs.com/niuu/p/10107212.html
# https://blog.csdn.net/orangerfun/article/details/128090128

import base64
import random
import string
try:
    # 老版包，安装命令为：pip install pycryptodome
    from Crypto.Cipher import AES
    old_version = True
except:
    # https://www.pycrypto.org/
    # TODO: 新版包
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    old_version = False
from binascii import b2a_hex, a2b_hex
from dramkit.gentools import isnull


EXPAND_STR = '\0'
DEFAULT_KEY = 'thisisdefaultkey'
DEFAULT_IV = 'thisisadefaultiv'


def expand(text, baselen=16, encoding='utf-8'):
    '''
    | 如果text不足minlen位的倍数就用空格补足
    | 注：utf-8编码中一个汉字占3个字节
    '''
    n = len(text.encode(encoding))
    if n % baselen:
        n_ = baselen - (n % baselen)
        text = text + (EXPAND_STR * n_)
    return text.encode(encoding)


def _check_key(key, encoding='utf-8'):
    return expand(DEFAULT_KEY if isnull(key) else key,
                  encoding=encoding)


def _check_iv(iv, encoding='utf-8'):
    return expand(DEFAULT_IV if isnull(iv) else iv,
                  encoding=encoding)


def _en_aes_cbc_old(btext, bkey, biv):
    mode = AES.MODE_CBC    
    cryptos = AES.new(bkey, mode, biv)
    cipher_text = cryptos.encrypt(btext)
    return cipher_text


def _en_aes_cbc_new(btext, bkey, biv):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(bkey),
                    modes.CBC(biv),
                    backend=backend)
    encryptor = cipher.encryptor()
    cipher_text = encryptor.update(btext)
    return cipher_text


def en_aes_cbc(text, key=None, iv=None, encoding='utf-8'):
    '''AES CBC加密'''
    bkey = _check_key(key, encoding=encoding)
    biv = _check_iv(iv, encoding=encoding)
    btext = expand(text, encoding=encoding)
    if old_version:
        cipher_text = _en_aes_cbc_old(btext, bkey, biv)
    else:
        cipher_text = _en_aes_cbc_new(btext, bkey, biv)
    # 因为AES加密后的字符串不一定是ascii字符集的，输出保存可能存在问题，
    # 所以这里转为16进制字符串
    return b2a_hex(cipher_text).decode(encoding)


def _de_aes_cbc_old(text, bkey, biv):
    mode = AES.MODE_CBC
    cryptos = AES.new(bkey, mode, biv)
    plain_text = cryptos.decrypt(a2b_hex(text))
    return plain_text


def _de_aes_cbc_new(text, bkey, biv):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(bkey),
                    modes.CBC(biv),
                    backend=backend)
    decryptor = cipher.decryptor()
    plain_text = decryptor.update(a2b_hex(text))
    return plain_text


def de_aes_cbc(text, key=None, iv=None, encoding='utf-8'):
    '''AEC CBC解密'''
    bkey = _check_key(key, encoding=encoding)
    biv = _check_iv(iv, encoding=encoding)
    if old_version:
        plain_text = _de_aes_cbc_old(text, bkey, biv)
    else:
        plain_text = _de_aes_cbc_new(text, bkey, biv)
    return bytes.decode(plain_text, encoding).rstrip(EXPAND_STR)
    

def _en_aes_ecb_old(btext, bkey):
    mode = AES.MODE_ECB
    cryptos = AES.new(bkey, mode)
    cipher_text = cryptos.encrypt(btext)
    return cipher_text


def _en_aes_ecb_new(btext, bkey):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(bkey),
                    modes.ECB(),
                    backend=backend)
    encryptor = cipher.encryptor()
    cipher_text = encryptor.update(btext)
    return cipher_text


def en_aes_ecb(text, key=None, encoding='utf-8'):
    '''AES ECB加密'''
    bkey = _check_key(key, encoding=encoding)
    btext = expand(text, encoding=encoding)
    if old_version:
        cipher_text = _en_aes_ecb_old(btext, bkey)
    else:
        cipher_text = _en_aes_ecb_new(btext, bkey)
    return b2a_hex(cipher_text).decode(encoding)


def _de_aes_ecb_old(text, bkey):
    mode = AES.MODE_ECB
    cryptor = AES.new(bkey, mode)
    plain_text = cryptor.decrypt(a2b_hex(text))
    return plain_text


def _de_aes_ecb_new(text, bkey):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(bkey),
                    modes.ECB(),
                    backend=backend)
    decryptor = cipher.decryptor()
    plain_text = decryptor.update(a2b_hex(text))
    return plain_text


def de_aes_ecb(text, key=None, encoding='utf-8'):
    '''AES ECB加密'''
    bkey = _check_key(key, encoding=encoding)
    if old_version:
        plain_text = _de_aes_ecb_old(text, bkey)
    else:
        plain_text = _de_aes_ecb_new(text, bkey)
    return bytes.decode(plain_text, encoding).rstrip(EXPAND_STR)


def en_base64(text, encoding='utf-8'):
    '''base64加密'''
    btext = text.encode(encoding)
    base64_text = base64.b64encode(btext)
    return base64_text.decode(encoding)


def de_base64(text, encoding='utf-8'):
    '''base64解密'''
    btext = text.encode(encoding)
    source_text = base64.b64decode(btext)
    return source_text.decode(encoding)


def en_base64_url(text, encoding='utf-8'):
    '''base64加密'''
    btext = text.encode(encoding)
    base64_text = base64.urlsafe_b64encode(btext)
    return base64_text.decode(encoding)


def de_base64_url(text, encoding='utf-8'):
    '''base64解密'''
    btext = text.encode(encoding)
    source_text = base64.urlsafe_b64decode(btext)
    return source_text.decode(encoding)


def gen_random_token(length=64, spec_chars=False, rand_seed=None):
    selects = string.ascii_letters + string.digits
    if spec_chars:
        selects += string.punctuation
    random.seed(rand_seed)
    token = ''.join(random.choices(selects, k=length))
    return token


def gen_random_tokens(n, **kwargs):
    return [gen_random_token(**kwargs) for _ in range(n)]


if __name__ == '__main__':
    # '''
    encrypt, decrypt = en_aes_cbc, de_aes_cbc
    # encrypt, decrypt = en_aes_ecb, de_aes_ecb
    # encrypt, decrypt = en_base64, de_base64
    # encrypt, decrypt = en_base64_url, de_base64_url
    
    encoding = 'utf-8'
    # encoding = 'gbk'
    # encoding = 'ansi'
    
    o = '这是待加密字符串Aa1`~!@#$%^&*（）()_-+=|?/'
    e = encrypt(o, encoding=encoding)
    d = decrypt(e, encoding=encoding)
    print('原始:', o)
    print('加密:', e)
    print('解密:', d)
    print('一致:', o == d)
    # '''
    