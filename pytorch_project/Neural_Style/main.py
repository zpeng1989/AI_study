import torch as t
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
import torch.nn as nn


#from torchnet.meter import AverageValueMeter

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Config(object):
    image_size = 256
    batch_size = 8
    num_workers = 4 
    data_root = '/home/zhangp/Documents/data/'
    use_gpu = True
    
    style_path = 'style.jpg'
    lr = 1e-3
    env = 'neural-style'
    plot_every = 10

    epoches = 2

    content_weight = 1e5
    style_weight = 1e10

    #model_path = './checkpoints/0_style.pth'
    model_path = None
    debug_file = ''

    content_path = 'style.jpg'
    result_path = 'output.png'

opt = Config()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    #vis = utils.Visualizer(opt.env)
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
    #transformer = nn.DataParallel(transformer)
    transformer.to(device)

    vgg = Vgg16().eval()
    #vgg = nn.DataParallel(vgg)
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    #style = nn.DataParallel(style)
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
                for ft_y, gm_s in zip(features_y, gram_style):
                    gram_y = utils.gram_matrix(ft_y)
                    style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
                style_loss *= opt.style_weight

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                content_meter.add(content_loss.item())
                style_meter.add(style_loss.item())

                if (ii + 1)% opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()
                    #vis.plot('content_loss', content_meter.value()[0])
                    #vis.plot('style_loss', style_meter.value()[0])
                    #vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                    #vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
            #vis.save([opt.env])
            t.save(transformer.state_dict(), 'checkpoints/%s_1style.pth' % epoch)

#opt = Config()
#@t.no_grad()
#def stylize(**kwargs):
@t.no_grad()
def stylize1(**kwargs):
    opt = Config()
    #print(opt)
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    #print('sssssssssssss')
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()
    #print('TTTTTTTTTTTTTTTT')
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location = lambda _s, _: _s))
    #style_model = nn.DataParallel(style_model)
    style_model.to(device)

    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min = 0, max = 1), opt.result_path)


@t.no_grad()
def stylize(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire
    fire.Fire()




