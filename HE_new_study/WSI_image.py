# -*- coding: utf-8 -*-

### https://digitalslidearchive.github.io/HistomicsTK/examples/wsi-io-using-large-image.html#Download-a-sample-whole-slide-image

import large_image
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

#Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams['image.cmap'] = 'gray'

wsi_url = 'https://data.kitware.com/api/v1/file/5899dd6d8d777f07219fcb23/download'
wsi_path = 'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'

## 下载数据

if not os.path.isfile(wsi_path):
    prin(wsi_path)
    os.system("wget %s" %wsi_url)

ts = large_image.getTileSource(wsi_path)
## 载入数据
## 函数可以读取多种图像文件格式。它自动检测格式并提取它们之间的差异。有些格式需要安装可选依赖项
## 例如，需要安装large_image附加组件, 您的系统需要安装必要的核心包。可用的资源有
## 


print(ts.getMetadata())
## TileSource类的函数返回一个包含幻灯片基本元数据的python dict:

print(ts.getNativeMagnification())
## TileSource类的函数返回一个python dict，其中包含一个像素(以毫米为单位)在扫描幻灯片的基础或最高分辨率级别的放大倍数和物理大小(以毫米为单位)

print(ts.getMagnificationForLevel(level = 0))
## TileSource类的函数返回一个python dict，其中包含图像金字塔中指定级别的像素的放大倍数和物理大小。

for i in range(ts.levels):
    print('Level={}:{}'.format(i, ts.getMagnificationForLevel(level = i)))

print('Level with magnfication closest to 10x = {}'.format(ts.getLevelForMagnification(10)))
print('Level with pixel width closest to 0.0005mm = {}'.format(ts.getLevelForMagnification(mm_x = 0.0005)))
## TileSource类的函数返回与特定放大倍数或像素大小(以毫米为单位)相关的图像金字塔级别。


num_tiles = 0

tile_means = []
tile_areas = []
'''
for tile_info in ts.tileIterator(
        region= dict(left = 5000, top = 5000, width = 20000, height = 20000, units = 'base_pixels'),
        scale = dict(magnification = 20),
        tile_size = dict(width = 1000, height = 1000),
        tile_overlap = dict(x = 50, y = 50),
        format = large_image.tilesource.TILE_FORMAT_PIL
        ):
    ## 函数提供了一个迭代器，用于以块的方式按顺序迭代整个幻灯片或幻灯片中的感兴趣区域(ROI)，以获得所需的任何分辨率。
    if num_tiles % 100 == 0:
        print('Tile-{} ='.format(num_tiles))
        print(tile_info)
    im_tile = np.array(tile_info['tile'])
    tile_mean_rgb = np.mean(im_tile[:, :, :3], axis = (0,1))
    tile_means.append(tile_mean_rgb)
    print(tile_mean_rgb)
    tile_areas.append(tile_info['width'] * tile_info['height'])
    num_tiles += 1

slide_mean_rgb = np.average(tile_means, axis = 0, weights = tile_areas)

print('Number of tiles = {}'.format(num_tiles))
print('Sile mean color = {}'.format(slide_mean_rgb))

pos = 1000

tile_info = ts.getSingleTile(
        tile_size=dict(width=1000, height=1000),
        scale=dict(magnification=20),
        tile_position=pos
        )

pic_out = tile_info['tile']
cv2.imwrite('1.jpg',pic_out)



##函数的作用是:直接获取位于tile迭代器特定位置的tile。除了前面提到的tileIterator参数之外，它还接受一个tile_position参数，该参数可用于指定感兴趣的tile的线性位置。


'''

## getRegion() 函数可以通过以下两个参数在任意缩放/放大的情况下在幻灯片中获得感兴趣的矩形区域(ROI):

im_roi, _ = ts.getRegion(
        region = dict(left = 10000, top = 10000, width = 10000, height = 10000, units = 'base_pixels'),
        format = large_image.tilesource.TILE_FORMAT_NUMPY
)
## cv2.imwrite('2.jpg',im_roi)

im_low_res, _ = ts.getRegion(
        scale = dict(magnification = 1.25),
        format = large_image.tilesource.TILE_FORMAT_NUMPY
)

## cv2.imwrite('3.jpg', im_low_res)

## convertRegionScale() 函数可用于将区域从一个缩放/放大转换为另一个缩放/放大，如下例所示

tr = ts.convertRegionScale(
        sourceRegion = dict(left = 5000, top = 5000, width = 1000, height = 1000, units = 'mag_pixels'),
        sourceScale = dict(magnigication = 20),
        targetScale=dict(magnification=10),
        targetUnits='mag_pixels'
        #format = large_iamge.tilesource.TILE_FROMAT_NUMPY
)

print(tr)


## getRegionAtAnotherScale() 函数可用于获取另一尺度区域的图像。

im_roi, _ = ts.getRegionAtAnotherScale(
        sourceRegion = dict(left = 5000, top = 5000, width = 10000, height = 10000, units = 'mag_pixels'),
        targetScale = dict(magnification = 1.25),
        format = large_image.tilesource.TILE_FORMAT_NUMPY
)

print(im_roi.shape)

cv2.imwrite('4.jpg', im_roi)



