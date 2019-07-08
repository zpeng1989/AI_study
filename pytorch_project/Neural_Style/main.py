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
    transfroms = tv.transforms.Compose([
            tv.transforms.Resize(opt.image_size),
            tv.transforms.CenterCrop(opt.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: x * 255)
        ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location =lambda _s, _: _s))
    transformer.to(device)

    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = t.optim.Adam(transformer.parameters(), opt.lt)

    style = style.to(device)
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]


    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()
        for epoch in range(opt.epoches):
            content_meter.reset()
            style_meter.reset()
            for ii, (x,  _) in tqdm.tqdm(enumerate(dataloader)):
                optimizer.zero_grad()
                x = x.to(device)
                y = transformer(x)
                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)
                features_y = vgg(y)
                features_x = vgg(x)

                content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)
                style_loss = 0.





