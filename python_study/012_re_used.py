# coding: utf8
import re

def main():
    username = 'zpeng123'
    qq = 1234567890
    m1 = re.match(r'^[0-9a-zA-Z_]{6,20}$', username)
    if not m1:
        print('输入有效用户名')
    m2 = re.match(r'^[1-9]\d{4,11}$', str(qq))
    if not m2:
        print('输入有效QQ号')
    if m1 and m2:
        print('输入有效')


if __name__ == '__main__':
    main()


