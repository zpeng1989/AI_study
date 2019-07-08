import torch ad t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet

import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm 
import os
import ipdb

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Config(object):
    image_size = 256
    batch_size = 8
    data_root = ''
    num_workers = 4
    use_gpu = True
    
    style_path = 'style.jpg'
    lr = 1e-3
    env = 'neural-style'
    plot_every = 10

    epoches = 2

    content_weight = 1e5
    style_weight = 1e10

    model_path = None
    debug_file = ''

    content_path = 'input.png'
    result_path = 'output.png'

opt = Config()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils.Visualizer(opt.env)




