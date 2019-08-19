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


## Segment nuclei
## 得到对应核的通道
im_nuclei_stain = im_stains[:,:,0]
foreground_threashold = 60

## segment foreground
## 过滤
im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(im_nuclei_stain < foreground_threashold)


## 多尺度过滤
min_radius = 10
max_radius = 15

im_log_max, im_sigma_max = htk.filters.shape.cdog(
        im_nuclei_stain, im_fgnd_mask,
        sigma_min = min_radius * np.sqrt(2),
        sigma_max = max_radius * np.sqrt(2)
)

## detect and segment nuclei using local maximum clustering
## 聚类的种类

local_max_search_radius = 10

im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(im_log_max, im_fgnd_mask, local_max_search_radius)


# filter out small objects
min_nucleus_area = 80

im_nuclei_seg_mask = htk.segmentation.label.area_open(im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

# compute nuclei properties
objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

print('Number of nuclei = ', len(objProps))

segment_file = skimage.color.label2rgb(im_nuclei_seg_mask, im_input, bg_label=0)

cv2.imwrite('segment.jpg', segment_file)


for i in range(len(objProps)):
    c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
    width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
    height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1
    cur_bbox = {"type": "rectangle", "center": c, "width": width, "height": height}
    print(cur_bbox)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#%matplotlib inline

#Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams['image.cmap'] = 'gray'
titlesize = 24


plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(skimage.color.label2rgb(im_nuclei_seg_mask, im_input, bg_label=0), origin='lower')
plt.title('Nuclei segmentation mask overlay', fontsize=titlesize)

plt.subplot(1, 2, 2)
plt.imshow( im_input )
plt.xlim([0, im_input.shape[1]])
plt.ylim([0, im_input.shape[0]])
plt.title('Nuclei bounding boxes', fontsize=titlesize)

for i in range(len(objProps)):
    c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
    width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
    height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1

    cur_bbox = {
            "type":"rectangle",
            "center":c,
            "width":width,
            "height":height,
            }

    plt.plot(c[0], c[1], 'g+')
    mrect = mpatches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height], width, height, fill=False, ec='g', linewidth=2)
    plt.gca().add_patch(mrect)

plt.savefig('plot3.png', format='png')
plt.show


