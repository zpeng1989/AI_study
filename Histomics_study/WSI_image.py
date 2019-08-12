# -*- coding: utf-8 -*-

### https://digitalslidearchive.github.io/HistomicsTK/examples/wsi-io-using-large-image.html#Download-a-sample-whole-slide-image

import large_image
import os

import numpy as np
import matplotlib.pyplot as plt

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


print(ts.getMetadata())
print(ts.getNativeMagnification())
print(ts.getMagnificationForLevel(level = 0))

for i in range(ts.levels):
    print('Level={}:{}'.format(i, ts.getMagnificationForLevel(level = i)))

print('Level with magnfication closest to 10x = {}'.format(ts.getLevelForMagnification(10)))
print('Level with pixel width closest to 0.0005mm = {}'.format(ts.getLevelForMagnification(mm_x = 0.0005)))




num_tiles = 0

tile_means = []
tile_areas = []

for tile_info in ts.tileIterator(
        region= dict(left = 5000, top = 5000, width = 20000, height = 20000, units = 'base_pixels'),
        scale = dict(magnification = 20),
        tile_size = dict(width = 1000, height = 1000),
        tile_overlap = dict(x = 50, y = 50),
        format = large_image.tilesource.TILE_FORMAT_PIL
        ):
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




