## study_before

import pandas as pd
import numpy as np

data1 = [1,2,3,4,5,6]

arr1 = np.array(data1)
print(arr1)

arr2 = np.zeros(5)
arr3 = np.ones(5)
arr4 = np.empty(5)
arr5 = np.arange(5,dtype = np.float64)



print(arr2)
print(arr3)
print(arr4)
print(arr5)
print(arr5.dtype)



## transform type

arr = np.array(['1.12','2.12','1.75'],dtype = np.string_)
print(arr.dtype)
print(arr.astype(float))

