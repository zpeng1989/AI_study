import os
import ipdb
import torchvision as tv
import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter

class Config(object):
    data_path = ''
    num_workers = 4
    image_size = 96
    batch_size = 256
    max_epochs = 200
    lr1 = 2e-4
    lr2 = 2e-4
    beta1 = 0.5
    gpu = True
    nz = 100
    ngf = 64
    ndf = 64

    save_path = 'imgs/'

    vis = True
    env = 'GAN'
    plot_every = 20
    debug_file = ''
    d_every = 1
    g_every = 5
    save_every = 10
    netd_path = None
    netg_path = None

