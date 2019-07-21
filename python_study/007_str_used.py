def main():
    str1 = 'hello, world!'
    print(len(str1))
    print(str1.capitalize())
    print(str1.upper())
    print(str1.find('or'))
    print(str1.find('shit'))
    print('++++++++++++++111++++++++++++++')
    print(str1.startswith('He'))
    print(str1.startswith('hel'))
    print(str1.endswith('!'))
    print('++++++++++++++222++++++++++++++')
    print(str1.center(50, '*'))
    print(str1.rjust(50, ' '))
    str2 = 'abc123456'
    print(str2[2])
    print(str2[2:])
    print(str2[2::2])
    print(str2[::-1])
    str3 = '          jackfrued@126.com   '
    print(str3)
    print(str3.strip())


if __name__ == '__main__':
    main()



def main():
    list1 = [1, 3, 5, 7, 100]
    print(list1)
    list2 = ['hello'] * 5
    print(list2)
    print(len(list1))
    print(list1[0])
    print(list1[4])
    print(list1[-1])
    print(list1[-3])
    list1[2] = 300
    print(list1)



if __name__ == '__main__':
    main()







def main():
    fruits = ['grape', 'apple', 'strawberry', 'waxberry']
    fruits += ['pitaya', 'pear', 'mongo']
    for fruit in fruits:
        print(fruit.title())
        #print(fruit.title(), end=' ')
        #print(fruit.title(), end='')
    print()
    fruits2 = fruits[1:4]
    print(fruits2)
    fruits3 = fruits[:]
    print(fruits3)
    fruits4 = fruits[-3:-1]
    print(fruits4)
    fruits5 = fruits[::-1]
    print(fruits5)

if __name__ == '__main__':
    main()


def main():
    list1 = ['orange', 'apple', 'zoo', 'internationalization', 'blueberry']
    list2 = sorted(list1)
    list3 = sorted(list1, reverse = True)
    list4 = sorted(list1, key = len)
    print(list1)
    print(list2)
    print(list3)
    print(list4)

if __name__ == '__main__':
    main()


import sys

def main():
    f = [x for x in range(1, 10)]
    print(f)
    f = [x + y for x in 'ABCDEF' for y in '1234567']
    print(f)
    f = [x ** 2 for x in range(1, 1000)]
    print(sys.getsizeof(f))
    #print(f)
    f = (x ** 2 for x in range(1, 1000))
    print(sys.getsizeof(f))
    #print(f)



if __name__ == '__main__':
    main()



def main():
    t = ('zp', 38, True, 'shaanghai')
    print(t)
    print(t[0])
    for member in t:
        print(member)

    first_list = ['apple', 'banana', 'orange']
    fruits_tuple = tuple(first_list)
    print(fruits_tuple)


if __name__ == '__main__':
    main()



