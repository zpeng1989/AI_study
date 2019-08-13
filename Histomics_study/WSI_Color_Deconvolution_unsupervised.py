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


#inputImageFile = ('https://data.kitware.com/api/v1/file/','57802ac38d777f12682731a2/download')  # H&E.png

#imInput = skimage.io.imread(inputImageFile)[:, :, :3]
print(imInput_part.shape)

cv2.imwrite('12.jpg', imInput_part)

## 如果您知道用于颜色反褶积的染色矩阵是什么，那么计算反褶积图像就像在histomicstk.preprocessing.color_deconvolution中调用color_deconvolution函数一样简单。它的输入是一个RGB图像和染色矩阵，它的输出是一个namedtuple，其中包含字段污渍、

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('stain_colort_map:', stain_color_map, sep = '\n')
stains = ['hematoxylin',  # nuclei stain
            'eosin',      # cytoplasm stain
            'null']

W = np.array([stain_color_map[st] for st in stains]).T

W_init = W[:, :2]

# Compute stain matrix adaptively
sparsity_factor = 0.5

I_0 = 255
im_sda = htk.preprocessing.color_conversion.rgb_to_sda(imInput, I_0)
W_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(im_sda, W_init, sparsity_factor)

# perform sparse color deconvolution
imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
        imInput,
        htk.preprocessing.color_deconvolution.complement_stain_matrix(W_est),
        I_0
)

print('Estimated stain colors (in rows):', W_est.T, sep='\n')




for i in [0,1,2]:
    part_pic = imDeconvolved.Stains[:, :, i]
    output_path = stains[i] + 'part_output.jpg'
    cv2.imwrite(output_path, part_pic)




