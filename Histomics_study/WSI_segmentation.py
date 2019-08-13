# -*- coding: utf-8 -*-

import histomicstk as htk
import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import cv2

wsi_path = 'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
imInput = skimage.io.imread(wsi_path)[0]
im_input = imInput[5000:7000, 5000:7000, :3]

#print(imInput_part.shape)

## color normalization
## 色彩正则化
im_reference = imInput[7000:9000, 7000:9000, :3]

# 得到实验空间中参考图像的均值和stddev
mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

# 执行reinhard颜色归一化
im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)

cv2.imwrite('reference_image.jpg', im_reference)
cv2.imwrite('Normalized_output_image.jpg', im_nmzd)

## color deconvolution
## 色彩去卷积

stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin': [0.07, 0.99, 0.11],
        'dab': [0.27, 0.57, 0.78],
        'null': [0.0, 0.0, 0.0]
}

stain_1 = 'hematoxylin'   # nuclei stain
stain_2 = 'eosin'         # cytoplasm stain
stain_3 = 'null'          # set to null of input contains only two stains

W = np.array([stainColorMap[stain_1],
              stainColorMap[stain_2],
              stainColorMap[stain_3]]).T


im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains

cv2.imwrite('hemat.jpg', im_stains[:, :, 0])
cv2.imwrite('eosin.jpg', im_stains[:, :, 1])




