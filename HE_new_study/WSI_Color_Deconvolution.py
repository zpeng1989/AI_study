# -*- coding: utf-8 -*-

from __future__ import print_function

import histomicstk as htk
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

wsi_path = 'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'

imInput = skimage.io.imread(wsi_path)[0]

imInput_part = imInput[5000:7000, 5000:7000, :3]
print(imInput_part.shape)

cv2.imwrite('12.jpg', imInput_part)

## 如果您知道用于颜色反褶积的染色矩阵是什么，那么计算反褶积图像就像在histomicstk.preprocessing.color_deconvolution中调用color_deconvolution函数一样简单。它的输入是一个RGB图像和染色矩阵，它的输出是一个namedtuple，其中包含字段污渍、

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('stain_colort_map:', stain_color_map, sep = '\n')

stains = ['hematoxylin', 'eosin', 'null']


W = np.array([stain_color_map[st] for st in stains]).T

imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, W)

for i in [0,1,2]:
    part_pic = imDeconvolved.Stains[5000:6000, 5000:6000, i]
    output_path = stains[i] + 'sus_part_output.jpg'
    cv2.imwrite(output_path, part_pic)




