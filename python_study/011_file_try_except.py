import re


def main():
    username = 'zp'
    qq = 4353532432
    m1 = re.match(r'^[0-9a-zA-Z_]{6, 20}$', username)
    if not m1:
        print('输入有效')
