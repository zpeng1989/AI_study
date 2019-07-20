import os

x = float(4)

if x >1 :
    y = 3*x -5
elif x>= -1:
    y = x + 2
else:
    y = 5*x + 3

print(x , y)



x = float(32)

if x > 1:
    y = 3*x - 5
else:
    if x >= -1:
        y = x + 2
    else:
        y = 5 * x + 3
print(x, y)

## test 1

value = float(23)
unit = 'cm'

if unit == 'in':
    print(value *2.54)
elif unit == 'cm':
    print(value/ 2.54)
else:
    print('test')
