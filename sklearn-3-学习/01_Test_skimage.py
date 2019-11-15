## coding = utf-8

## https://www.jianshu.com/p/f2e88197e81d

import image as img
import os
#from matplotlib import pyplot as plot
import skimage
from skimage import io, transform, color, img_as_ubyte, img_as_float


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



