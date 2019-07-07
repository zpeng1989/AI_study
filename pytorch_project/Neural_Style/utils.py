from itertooks import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)/(ch * h * w)
    return gram

class Visualizer():
    def __init__(self, env = 'default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env = env, use_incoming_socker = False, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, use_incoming_socket = False, **kwargs)
        return self

    def plot_many(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y = np.array([y]),
                      X = np.array([x]),
                      win = name,
                      opts = dict(title = name),
                      update = None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y = np.array([y]),
                      X = np.array([x]),
                      win = name,
                      opts = dict(title = name)
        )
    
    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k,v)

    def img_grid(self, name, input_3d):
        self.img(name, tv.utils.make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max =1, min = 0)))

    def log(self, info, win = 'log_text'):
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self __getattr__(self, name):
            return getattr(self.vis, name)

def get_style_data(path):
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])
    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)






