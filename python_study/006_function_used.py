m = int(19)
n = int(4)

fm = 1
for num in range(1, m + 1):
    fm *= num

fn = 1
for num in range(1, n + 1):
    fn *= num

fmn = 1

for num in range(1, m - n + 1):
    fmn  *= num

print(fm // fn // fmn)


def factorial(num):
    result = 1
    for n in range(1, num + 1):
        result *= n
    return result

m = 19
n = 4

print(factorial(m)//factorial(n)//factorial(m - n))

from random import randint

def roll_dice(n = 2):
    total = 0
    for _ in range(n):
        total += randint(1, 6)
    return total


def add(a = 0, b = 0, c = 0):
    return a + b + c

print(roll_dice())
print(roll_dice(3))

print(add())
print(add(1))
print(add(1, 2))
print(add(1, 2, 3))


def add(*args):
    total = 0
    for val in args:
        total += val
    return total

print(add())
print(add(1))
print(add(1, 2))
print(add(1, 2, 3))




def gcd(x, y):
    (x, y) = (y, x) if x > y else (x, y)
    for factor in range(x, 0, -1):
        if x % factor == 0 and y % factor == 0:
            return factor

def lcm(x, y):
    return x * y //gcd(x, y)


print(lcm(100, 26))
print(gcd(100, 26))


def is_prime(num):
    for factor in range(2, num):
        if num % factor == 0:
            return False
    return True if num != 1 else False



def foo():
    b = 'hello'
    def bar():
        c = True
        print(a)
        print(b)
        print(c)
    bar()

if __name__ == '__main__':
    a = 100 
    print(foo())





