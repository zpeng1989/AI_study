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


from random import randint
face = randint(1,6)

if face == 1:
    result = 'song'
elif face == 2:
    result = 'dancer'
elif face == 3:
    result = 'dog'
elif face == 4:
    result = 'sport'
elif face == 5:
    result = 'kou'
elif face == 6:
    result = 'joke'

print(result)


# SciClone was employed to analyze the clonal structure, based on a Bayesian clustering method. 
# 
# An independent input was used to analyze the clonal structure in Primary Tumor and the Metastatic Tumor for DNA at baseline and matched tissue samples, respectively. 
# 
# For serial DNA, multiple inputs of each sample were used to analyze serial clonal population. 
# 
# Cancer cell fraction was calculated with the mean of predicted cellular frequencies. 
# 
# The cluster with the highest mean VAF was identified as the clonal cluster, and mutations in this cluster were clonal mutations.
