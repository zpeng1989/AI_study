# -*- coding: utf-8 -*-

from __future__ import print_function
import large_image
import matplotlib.pyplot as plt
import skimage.io
import cv2
import histomicstk.segmentation.positive_pixel_count as ppc

#Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 15, 15
plt.rcParams['image.cmap'] = 'gray'
wsi_path = 'PAD.png'
imInput = skimage.io.imread(wsi_path)
im_input = imInput

#print(imInput_part.shape)
## color normalization
## 色彩正则化

def count_and_label(im_input, params, im_output):
    #"Compute the label image with count_image, and then display it"
    label_image = ppc.count_image(im_input, params)[1]
    #cv2.imwrite(im_output, label_image)
    plt.imshow(label_image)
    #plt.show()
    plt.savefig(im_output, format='png')


template_params = ppc.Parameters(
                    hue_value=0.05,
                    hue_width=0.15,
                    saturation_minimum=0.05,
                    intensity_upper_limit=0.95,
                    intensity_weak_threshold=0.65,
                    intensity_strong_threshold=0.35,
                    intensity_lower_limit=0.05,
                    )

im_output = 'output_v1.png'
count_and_label(im_input, template_params, im_output)

#stats, label_image = ppc.count_image(im_input)

stats, label_image = ppc.count_image(im_input, template_params)

## 在这里，我们看到一些以前被认为是正的像素现在被认为是负的，在可见的核中产生了洞。这表明我们将色调范围移动得太远了。
## 无论如何，我们也可以查看统计数据。这里我们使用原始的参数值。

def pp_namedtuple(t):
    # "Pretty-print a namedtuple by printing each field on its own line and left-aligning all values"
    print(type(t).__name__)
    maxlen = max(map(len, t._fields))
    for f in t._fields:
        print(f, getattr(t, f), sep=':' + ' ' * (maxlen - len(f)) + '\t')

pp_namedtuple(stats)


slide_path = 'TCGA-DX-A6BG-01Z-00-DX2.34763958-0613-4069-9ACC-13D6633FE415.svs'
region = dict(left=50000, top=35000, width=1600, height=900)
ts = large_image.getTileSource(slide_path)
im_region = ts.getRegion(region=region, format=large_image.tilesource.TILE_FORMAT_NUMPY)[0]

import cv2

stats, label_image = ppc.count_slide(slide_path, template_params, region=region, make_label_image=True)
pp_namedtuple(stats)
cv2.imwrite('a1.jpg',label_image)
plt.imshow(label_image)
#plt.show()
plt.savefig('aa1.png', format='png')


stats_dask, label_image = ppc.count_slide(slide_path, template_params, region=region, tile_grouping=1, make_label_image=True)
pp_namedtuple(stats_dask)

plt.imshow(label_image)
plt.show()
plt.savefig('aa3.png', format='png')


large_region = dict(left=6e3, top=3e3, width=3e3, height=3e3)
stats, label_image = ppc.count_slide(slide_path, template_params, large_region, make_label_image=True)
pp_namedtuple(stats)
plt.imshow(label_image)
plt.show()
plt.savefig('aa2.png', format='png')


cv2.imwrite('a2.jpg', label_image)

large_region = dict(left=60e3, top=30e3, width=30e3, height=30e3)
stats, = ppc.count_slide(slide_path, template_params, large_region)

pp_namedtuple(stats)


### describe
## Number Weak Positive
## 弱阳性
## Number Positive
## 阳性
## Number Strong Positive
## 强阳性
## Intensity Sum Weak Positive
## 强 弱阳性
## Intensity Sum Positive
## 强 阳性
## Intensity Sum Strong Positive
## 强 强阳性
## Intensity Average
## 强 平均值
## Ratio Strong To Total
## 阳性占比
## Intensity Average Weak And Positive
## 强阳性占比哦



