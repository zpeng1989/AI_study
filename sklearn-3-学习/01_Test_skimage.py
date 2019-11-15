## coding = utf-8

## https://www.jianshu.com/p/f2e88197e81d

import image as img
import os
#from matplotlib import pyplot as plot
import skimage
from skimage import io, transform, color, img_as_ubyte, img_as_float
import numpy as np

#img_file1 = img.open('2.png')
img_file2 = io.imread('2.png')

print("the picture's shape: ", img_file2.shape)


print(img_file2[550][550])
print(img_file2[500][1000])

print(img_file2.shape)
print(img_file2.size)
print(img_file2.max())
print(img_file2.min())
print(img_file2.mean())


img_gray = color.rgb2gray(img_file2)
rows,cols = img_gray.shape

for i in range(rows):
	for j in range(cols):
		if (img_gray[i, j] <= 0.5):
			img_gray[i, j] = 0
		else:
			img_gray[i, j] = 1

io.imsave('2p.png',img_gray)


#############################将三通道转为灰度图


img_file22=skimage.color.rgb2gray(img_file2)
print(type(img_file22),img_file22.shape,img_file22.dtype,img_file22.max(),img_file22.min(),img_file22.mean() )
dst=img_as_ubyte(img_file22)
print(type(dst),dst.shape,dst.dtype, dst.max(), dst.min(), dst.mean())



############################## 查看数据类型

print(img_file2.dtype.name)
dst = img_as_float(img_file2)
print(dst.dtype.name)


############################ 颜色空间之间转化

# rgb cie', 'xyz', 'yuv', 'yiq', 'ypbpr', 'ycbcr', 'ydbdr
hsv = color.convert_colorspace(img_file2[:,:,0:3], 'RGB', 'HSV')
#hsv = color.convert_colorspace(img_file2, 'rgb cie', 'HSV')
#print('+++++++++')
#hsv = color.convert_colorspace(img_file2, 'xyz', 'HSV')
#hsv = color.convert_colorspace(img_file2, 'yiq', 'HSV')
#hsv = color.convert_colorspace(img_file2, 'ypbpr', 'HSV')
#hsv = color.convert_colorspace(img_file2, 'ycbcr', 'HSV')
#hsv = color.convert_colorspace(img_file2, 'ydbdr', 'HSV')


io.imsave('2p1.png',hsv)
io.imsave('2p2.png',img_file2)
pic_one = img_file2[:,:,0]
pic_two = img_file2[:,:,1]
pic_thr = img_file2[:,:,2]
pic_for = img_file2[:,:,3]


io.imsave('p_1.png',pic_one)
io.imsave('p_2.png',pic_two)
io.imsave('p_3.png',pic_thr)
#io.imsave('p_4.png',pic_for)

print(type(img_file2))

### -----------------
### 修改大小 resize()
### -----------------
### skimage.transform.resize(image, output_shape)

from skimage import transform, data
img = data.camera()
dst = transform.resize(img, (80, 60))

### ----------------
### 按照比例修改 rescale()
### ----------------

print(img.shape)
print(transform.rescale(img, 0.1).shape)
print(transform.rescale(img, [0.5, 0.25]).shape)
print(transform.rescale(img, 2).shape)

### -----------------
### 旋转 rotate
### -----------------


from skimage import transform, data

img = data.camera()
print(img.shape)

img1 = transform.rotate(img, 60)
print(img1.shape)

img2 = transform.rotate(img, 30, resize = True)
print(img2.shape)


#####################
# 对比对增强 exposure
#####################
# skimage.exposure.adjust_gamma(image, gamma = 1)

from skimage import data, exposure, img_as_float
image = img_as_float(data.moon())
gam1 = exposure.adjust_gamma(image, 2)
gam2 = exposure.adjust_gamma(image, 0.5)

io.imsave('out1.png', gam1)
io.imsave('out1_2.png', gam2)

from skimage import data, exposure
image = data.moon()
result = exposure.is_low_contrast(image)

print(result)

####################
# 拉伸 skimage.exposure.rescale_intensity(image, in_range = 'image', out_range = 'dtype')
###################
image = np.array([51, 102, 153], dtype = np.uint8)
mat = exposure.rescale_intensity(image)
print(mat)

tmp = image * 1.0
mat = exposure.rescale_intensity(tmp)
print(mat)


image = np.array([-10, 0, 10], dtype = np.int8)
mat = exposure.rescale_intensity(image, out_range = (0, 127))
print(mat)



#####################
# 绘制直方图 skimage.exposure.histogram(image, nbins = 256)
#####################

image = data.camera()*1.0

hist1 = np.histogram(image, bins = 2)

print(hist1)
#io.imsave('his.png', hist1)

import matplotlib.pyplot as plt

img=data.camera()
arr=img.flatten()
print('KKKKKKKKKKKKKKKKKKKKK')
print(arr)
print('IIIIIIIIIIIIIIIIIIIII')
print(arr.min())
print(arr.max())
print(arr.mean())

#n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red') 
#print(n)
#print(bins)
#print(patches)



img = io.imread('2.png')
ar=img[:,:,0].flatten()
ag=img[:,:,1].flatten()
ab=img[:,:,2].flatten()

print(ar.mean())
print(ag.mean())
print(ab.mean())























