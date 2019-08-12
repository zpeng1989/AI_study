import large_image
import os

import numpy as np
import matplotlib.pyplot as plt

#Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams['image.cmap'] = 'gray'

wsi_url = 'https://data.kitware.com/api/v1/file/5899dd6d8d777f07219fcb23/download'
wsi_path = 'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'


if not os.path.isfile(wsi_path):
    print(wsi_path)
    os.system("wget %s" %wsi_url)

ts = large_image.getTileSource(wsi_path)

print(ts.getMetadata())
print(ts.getNativeMagnification())






